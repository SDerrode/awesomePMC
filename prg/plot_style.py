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


# Apply on import.
apply_style()
