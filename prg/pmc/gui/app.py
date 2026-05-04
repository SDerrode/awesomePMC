"""
app.py — QApplication entry point for the PMC GUI.

Usage
-----
  # From the command line:
  python -m prg.pmc gui [MODEL.toml]

  # From Python:
  from prg.pmc.gui import run_gui
  run_gui()
  run_gui("my_model.toml")

Logging
-------
The GUI sets up file logging to ~/.copulasformm/pmc.log (DEBUG level) and
attaches a QTextEdit handler so every log record from the prg.* namespace
also appears in the GUI's log panel.
"""

import logging
import sys


def run_gui(model_path: str | None = None) -> int:
    """
    Start the PMC GUI application.

    Parameters
    ----------
    model_path : str, optional
        TOML model to load at startup.

    Returns
    -------
    int — exit code (0 = clean exit).
    """
    # ── Logging: file handler first (before any prg import that logs) ────
    from prg.pmc.logging_setup import configure, add_widget_handler
    log_file = configure(level=logging.WARNING)   # console: WARNING+; file: DEBUG+

    # ── Qt application ───────────────────────────────────────────────────
    from PyQt6.QtWidgets import QApplication
    from prg.pmc.gui.main_window import PMCMainWindow

    app    = QApplication.instance() or QApplication(sys.argv)
    window = PMCMainWindow(model_path=model_path)

    # Attach QTextEdit handler *after* the window (and its log panel) exist
    add_widget_handler(window._log, level=logging.INFO)

    if log_file:
        window._log.append(f"[logging] File: {log_file}")
    window._log.append("[logging] Level: file=DEBUG  panel=INFO  console=WARNING")

    window.show()
    return app.exec()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(run_gui(path))
