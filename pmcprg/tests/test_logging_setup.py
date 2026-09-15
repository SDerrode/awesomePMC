"""
test_logging_setup.py — coverage for ``pmcprg.pmc.logging_setup``.

Covers:

* ``configure(log_file=False)``   — file logging disabled.
* ``configure(log_file=...)``     — explicit path; round-trip through
  the file handler.
* Re-entry guard (``_configured`` flag).
* OSError path (unwriteable log file → warning, no crash).
* ``add_widget_handler``          — Qt widget receives formatted records
  (skipped if PyQt6 is not installed).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from pmcprg.pmc import logging_setup


# ---------------------------------------------------------------------------
# Fixture — reset the module-level ``_configured`` flag and any handlers we
# may have added, so each test starts from a clean slate.
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_logger():
    """Tear down any handler the logger already has, run the test, restore."""
    root = logging.getLogger("pmcprg")
    saved_handlers = list(root.handlers)
    saved_level    = root.level
    saved_flag     = logging_setup._configured

    for h in saved_handlers:
        root.removeHandler(h)
    logging_setup._configured = False

    try:
        yield root
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
        for h in saved_handlers:
            root.addHandler(h)
        root.setLevel(saved_level)
        logging_setup._configured = saved_flag


# ---------------------------------------------------------------------------
# configure()
# ---------------------------------------------------------------------------

def test_configure_no_file(fresh_logger):
    """``log_file=False`` disables file logging and returns None."""
    out = logging_setup.configure(level=logging.INFO, log_file=False)
    assert out is None
    # Only the stream handler should be attached
    handlers = fresh_logger.handlers
    assert len(handlers) == 1
    assert isinstance(handlers[0], logging.StreamHandler)


def test_configure_explicit_path(fresh_logger, tmp_path: Path):
    """Explicit log_file path: handler attached, file written, path returned."""
    log_path = tmp_path / "subdir" / "test.log"   # parent does not yet exist
    out = logging_setup.configure(level=logging.WARNING, log_file=log_path)

    assert out == log_path
    assert log_path.exists()

    # Emit a record and check it lands in the file
    logging.getLogger("pmcprg.pmc.test").error("hello world")
    for h in fresh_logger.handlers:
        h.flush()
    text = log_path.read_text()
    assert "hello world" in text


def test_configure_is_idempotent(fresh_logger, tmp_path: Path):
    """Calling configure() twice is a no-op on the second call."""
    log_path = tmp_path / "first.log"
    first  = logging_setup.configure(log_file=log_path)
    second = logging_setup.configure(log_file=tmp_path / "second.log")
    assert first  == log_path
    assert second is None
    # Second-call file should NOT have been created
    assert not (tmp_path / "second.log").exists()


def test_configure_unwriteable_path(fresh_logger):
    """A path that cannot be opened logs a warning, returns None, no crash."""
    # ``/proc/...`` exists on Linux, doesn't on macOS — use a guaranteed bad
    # path: a regular file used as if it were a directory.
    bad = Path("/dev/null") / "this-cannot-be-written" / "x.log"
    out = logging_setup.configure(log_file=bad)
    assert out is None
    # Console handler still attached even though file failed
    has_stream = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in fresh_logger.handlers
    )
    assert has_stream


# ---------------------------------------------------------------------------
# add_widget_handler — requires PyQt6
# ---------------------------------------------------------------------------

PyQt6 = pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication, QTextEdit   # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def test_widget_handler_receives_records(fresh_logger, tmp_path: Path, qapp):
    """A QTextEdit attached via add_widget_handler accumulates formatted records."""
    logging_setup.configure(log_file=tmp_path / "x.log")
    edit = QTextEdit()
    handler = logging_setup.add_widget_handler(edit, level=logging.DEBUG)
    try:
        logging.getLogger("pmcprg.pmc.test").info("message-from-test")
        for h in fresh_logger.handlers:
            h.flush()
        # Allow Qt to deliver the queued signal synchronously by spinning events
        qapp.processEvents()
        assert "message-from-test" in edit.toPlainText()
    finally:
        fresh_logger.removeHandler(handler)
