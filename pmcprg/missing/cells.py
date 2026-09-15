"""pmcprg.missing.cells — how a missing observation is spelled in a CSV cell.

Shared by the CLI (``pmcprg.pmc.cli._read_observations``) and GUI
(``pmcprg.pmc.gui.main_window._read_data_csv``) loaders, so both read the same
files the same way.
"""

from __future__ import annotations

__all__ = ["MISSING_TOKENS", "parse_cell"]

#: Cell contents read as a missing value, compared after ``strip().lower()``:
#: an empty cell, ``NaN`` / ``nan`` (any case) and ``NA`` (R's spelling).
MISSING_TOKENS = frozenset({"", "nan", "na"})


def parse_cell(cell) -> float:
    """``float(cell)``, or NaN when the cell spells a missing value.

    Anything else that ``float`` rejects — text, or ``None`` for a row short
    of fields — raises ``ValueError`` / ``TypeError`` for the caller to turn
    into a row-aware message. ``inf`` parses as infinity; loaders decide
    whether to accept it.
    """
    if isinstance(cell, str) and cell.strip().lower() in MISSING_TOKENS:
        return float("nan")
    return float(cell)
