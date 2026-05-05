"""
prg.plot_style — package-wide matplotlib defaults.

Importing this module mutates ``matplotlib.rcParams`` so that every plot in
the package shares the same look (white facecolor, sane DPI, slightly
larger fonts than the matplotlib default).

Idempotent: re-importing has no extra effect (rcParams is just overwritten
with the same values).

Usage
-----
The plotting code in ``prg.copulas`` imports this once at package init.
End-users wanting to customise styling can either:
  * call ``apply_style(font_size=14, dpi=200)`` after importing prg, or
  * mutate ``matplotlib.rcParams`` directly afterwards (last-write-wins).
"""

import matplotlib as _mpl


# Default values — mirror the legacy ``prg.settings.plot_settings`` constants.
DEFAULT_FACECOLOR = "white"
DEFAULT_DPI       = 150
DEFAULT_FONT_SIZE = 12


def apply_style(
    facecolor: str = DEFAULT_FACECOLOR,
    dpi:       int = DEFAULT_DPI,
    font_size: int = DEFAULT_FONT_SIZE,
) -> None:
    """Update ``matplotlib.rcParams`` with the package-wide defaults.

    Parameters
    ----------
    facecolor : str — figure & savefig facecolor (default 'white').
    dpi       : int — figure & savefig DPI (default 150).
    font_size : int — base font size used for body text, axes labels AND
                      titles (default 12). Code that wants larger emphasis
                      passes ``fontsize=DEFAULT_FONT_SIZE + 2`` explicitly.
    """
    _mpl.rcParams.update({
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
    })


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


# Apply on import.
apply_style()
