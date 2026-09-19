"""Shared pytest setup for the awesomePMC test suite.

The suite historically assumed it was invoked from the repository root, so
the many module-level ``Path("pmcprg/pmc/models")`` constants and literal model
paths — and ``import pmcprg`` itself — resolve. ``pyproject.toml``'s
``testpaths`` and both CI configs run from the root, so this held in practice,
but ``pytest pmcprg/tests/test_sem.py`` from any other directory raised confusing
``FileNotFoundError`` / ``ModuleNotFoundError`` (audit T-2).

This conftest anchors both, independent of the invoking cwd:

* It prepends the repository root to ``sys.path`` at collection time so
  ``import pmcprg`` works even without an editable install.
* An autouse, session-scoped fixture runs the whole session from the repo
  root, so the cwd-relative model paths resolve. Tests that spawn subprocesses
  pass an explicit ``cwd=`` (and, for the CLI tests, ``PYTHONPATH``) and are
  therefore unaffected by this chdir.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Make ``import pmcprg`` resolve from this checkout regardless of install state.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True, scope="session")
def _run_from_repo_root():
    """Run the whole session from the repo root so cwd-relative paths resolve."""
    prev = Path.cwd()
    os.chdir(REPO_ROOT)
    try:
        yield
    finally:
        os.chdir(prev)


@pytest.fixture(autouse=True, scope="session")
def _qt_application():
    """One ``QApplication`` for the whole session, created before any test.

    ``pmcprg.pmc.gui.main_window`` switches matplotlib to the ``QtAgg`` backend
    when it is first imported, and matplotlib refuses that backend on a
    headless Linux (no display) unless a ``QApplication`` already exists — as
    it always does when the GUI really starts. A test that imports the window
    *before* building an application therefore passed or failed depending on
    which test ran first in its xdist worker: green on every push (the fast
    suite happens to import it after another test built one), red on Python
    3.13 in the full suite, where ``test_estim_missing`` came first. Building
    the application here removes the order dependence for every GUI test.
    Without PyQt6 (the ``gui`` extra is optional) it does nothing.
    """
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        yield None
        return
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def venv_kernel_manager():
    """A Jupyter ``KernelManager`` pinned to *this* interpreter's ipykernel.

    ``NotebookClient(..., kernel_name="python3")`` resolves the kernel through
    the standard Jupyter search path, where a user-level kernelspec (e.g.
    ``~/Library/Jupyter/kernels/python3``) shadows the venv's own kernel and
    can point at a different interpreter entirely — making the notebook tests
    depend on the machine's Jupyter state instead of on this venv (audit T-8).
    Restricting the kernelspec search to an empty dir list forces
    ``jupyter_client``'s native-kernel fallback: the spec ``ipykernel`` builds
    for ``sys.executable``.
    """
    kernelspec = pytest.importorskip("jupyter_client.kernelspec")
    manager    = pytest.importorskip("jupyter_client.manager")
    ksm = kernelspec.KernelSpecManager(kernel_dirs=[])
    return manager.KernelManager(kernel_name="python3", kernel_spec_manager=ksm)
