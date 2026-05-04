"""
logging_setup.py — Centralised logging configuration for prg.pmc.

Usage
-----
  # CLI (already called by cli.main):
  from prg.pmc.logging_setup import configure
  configure(level=logging.DEBUG, log_file=Path("run.log"))

  # GUI (called in app.py before showing the window):
  from prg.pmc.logging_setup import configure, add_widget_handler
  configure()
  add_widget_handler(my_qtextedit_widget)

Log file location
-----------------
Default: ~/.copulasformm/pmc.log  (created automatically).
Override by passing log_file=Path(...) to configure().
Pass log_file=False to disable file logging entirely.

Record format (file)
--------------------
  2026-05-04 12:34:56 DEBUG    prg.pmc.ice — _neg_wll: copula eval failed …
  2026-05-04 12:34:56 WARNING  prg.pmc.inference — Forward: C_1=0 at n=0 …

Thread safety
-------------
The console and file handlers are thread-safe (Python logging uses internal locks).
The QTextEdit handler routes records through a Qt signal/slot pair so it is also
safe to call from background QThread workers.
"""

import logging
import sys
from pathlib import Path

# Root logger for the whole prg namespace
_PRG_LOGGER = logging.getLogger("prg")

# File formatter — timestamp + level + module + full message (including exc_info tracebacks)
_FILE_FMT = logging.Formatter(
    fmt="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Console formatter — compact
_CONSOLE_FMT = logging.Formatter(
    fmt="%(levelname)s %(name)s — %(message)s",
)

_configured = False


def configure(
    level: int = logging.WARNING,
    log_file: "Path | bool | None" = None,
    file_level: int = logging.DEBUG,
) -> "Path | None":
    """
    Configure the ``prg`` logger with a console handler and a file handler.

    Parameters
    ----------
    level      : int   Console log level (default WARNING).
    log_file   : Path  Log file path. Defaults to ``~/.copulasformm/pmc.log``.
                       Pass ``False`` to disable file logging entirely.
    file_level : int   File log level (default DEBUG — captures full tracebacks).

    Returns
    -------
    Path | None  Resolved log-file path, or None if file logging is disabled.
    """
    global _configured

    root = _PRG_LOGGER
    root.setLevel(logging.DEBUG)   # handlers control the effective level

    # Guard against double-configuration (e.g. CLI then GUI in same process)
    if _configured:
        return None

    # ── Console handler ───────────────────────────────────────────────────
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(level)
    ch.setFormatter(_CONSOLE_FMT)
    root.addHandler(ch)

    # ── File handler ──────────────────────────────────────────────────────
    resolved: "Path | None" = None
    if log_file is not False:
        if log_file is None:
            log_file = Path.home() / ".copulasformm" / "pmc.log"
        log_file = Path(log_file)
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(file_level)
            fh.setFormatter(_FILE_FMT)
            root.addHandler(fh)
            resolved = log_file
            root.info("Log file opened: %s", resolved)
        except OSError as exc:
            root.warning(
                "Cannot open log file %s: %s — file logging disabled.",
                log_file, exc,
            )

    _configured = True
    return resolved


def add_widget_handler(widget, level: int = logging.INFO) -> "_QTextEditHandler":
    """
    Attach a handler that writes log records to a ``QTextEdit`` widget.

    Thread-safe: records emitted from background ``QThread`` workers are
    routed through a Qt queued signal so the actual ``widget.append()`` call
    always runs on the GUI thread.

    Must be called **after** the ``QApplication`` has been created.

    Parameters
    ----------
    widget : QTextEdit  Target widget (must expose an ``append(str)`` slot).
    level  : int        Minimum level forwarded to the widget (default INFO).

    Returns
    -------
    _QTextEditHandler  The installed handler (can be used to remove it later).
    """
    handler = _QTextEditHandler(widget, level=level)
    _PRG_LOGGER.addHandler(handler)
    return handler


# ---------------------------------------------------------------------------
# Thread-safe QTextEdit logging handler
# ---------------------------------------------------------------------------

class _LogEmitter:
    """
    A tiny QObject-derived signal carrier.

    Defined lazily so that importing ``logging_setup`` does NOT require
    PyQt6 to be installed (the module is also used in CLI mode).
    """
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            from PyQt6.QtCore import QObject, pyqtSignal

            class _Emitter(QObject):
                message = pyqtSignal(str)

            cls._instance = _Emitter()
        return cls._instance


class _QTextEditHandler(logging.Handler):
    """
    Thread-safe logging handler that appends formatted records to a QTextEdit.

    Records from background threads are delivered via a Qt queued connection,
    so ``widget.append()`` is always called on the GUI thread.
    """

    def __init__(self, widget, level: int = logging.INFO):
        super().__init__(level)
        self._widget  = widget
        self._emitter = _LogEmitter.get()
        # Connect once — multiple handlers share the same emitter instance
        # but each wraps its own widget, so use a lambda closure.
        self._emitter.message.connect(
            lambda msg, w=widget: w.append(msg),
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._emitter.message.emit(msg)
        except Exception:
            self.handleError(record)
