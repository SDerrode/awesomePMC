"""
pmcprg.pmc.gui.worker — generic background worker for the PMC GUI.

Runs heavy computation off the GUI thread and emits ``done(object)`` on
success / ``error(str)`` on failure (with a full traceback).

Optionally also emits ``progress(int, int, float, str)`` for callers that
want to plumb fine-grained status updates back to the UI thread (used by
the ICE estimate path to drive the determinate progress bar).
"""

from __future__ import annotations

import traceback

from PyQt6.QtCore import QThread, pyqtSignal


class _Worker(QThread):
    """Generic worker that calls ``func(*args, **kwargs)`` and emits the result.

    Signals
    -------
    done(object) : emitted with the return value of ``func`` on success.
    error(str)   : emitted with ``traceback.format_exc()`` on any exception.
    progress(int, int, float, str)
        Emitted by tasks that opt in. Arguments mirror the ICE
        ``progress_cb`` signature: ``(iter, max_iter, log_lik_or_value, run_tag)``.
        Tasks that don't track a log-likelihood (e.g. GoF) pass ``0.0`` for the
        third argument and use ``run_tag`` to convey the per-step label.
    """

    done     = pyqtSignal(object)
    error    = pyqtSignal(str)
    progress = pyqtSignal(int, int, float, str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self._func   = func
        self._args   = args
        self._kwargs = kwargs

    # ------------------------------------------------------------------
    # Progress plumbing
    # ------------------------------------------------------------------

    # Convenience callable suitable as a ``progress_cb`` argument.
    # Bound method → can be passed across threads safely; the signal
    # itself crosses the thread boundary (Qt::QueuedConnection by default).
    def emit_progress(
        self, it: int, max_iter: int, value: float = 0.0, run_tag: str = ""
    ) -> None:
        self.progress.emit(int(it), int(max_iter), float(value), str(run_tag))

    def set_progress_cb(self, kwarg_name: str = "progress_cb") -> None:
        """Inject :meth:`emit_progress` into the worker's kwargs.

        Use this rather than mutating ``_kwargs`` from the host window — it
        keeps the storage detail private to ``_Worker``.

        ``kwarg_name`` lets callers target a function that uses a different
        keyword argument (defaults to ``progress_cb``, the ICE convention).
        """
        self._kwargs[kwarg_name] = self.emit_progress

    # ------------------------------------------------------------------
    # QThread API
    # ------------------------------------------------------------------

    def run(self):
        try:
            result = self._func(*self._args, **self._kwargs)
            self.done.emit(result)
        except Exception:
            self.error.emit(traceback.format_exc())
