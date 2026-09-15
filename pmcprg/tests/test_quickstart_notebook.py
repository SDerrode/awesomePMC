"""
test_quickstart_notebook.py — execute the demo notebook end-to-end as a
regression test.

The notebook lives at ``examples/quickstart.ipynb``. We execute every code
cell with a fresh kernel and fail the test if any cell raises. Skipped if
``nbformat`` / ``nbclient`` are not installed (they are not in the hard
deps; they are pulled in by anyone who runs the notebook locally).

Tagged ``slow`` because it spins up a Jupyter kernel and runs ICE +
matplotlib plots (~10 s).
"""

from __future__ import annotations

from pathlib import Path

import pytest


nbformat  = pytest.importorskip("nbformat")
nbclient  = pytest.importorskip("nbclient")
ipykernel = pytest.importorskip("ipykernel")   # noqa: F401  (kernel runtime)


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK  = REPO_ROOT / "examples" / "quickstart.ipynb"


@pytest.mark.slow
def test_quickstart_notebook_executes(venv_kernel_manager):
    """Every code cell of examples/quickstart.ipynb must run without error."""
    assert NOTEBOOK.exists(), f"missing notebook: {NOTEBOOK}"

    nb = nbformat.read(NOTEBOOK, as_version=4)
    # km= pins the kernel to this interpreter — a bare kernel_name="python3"
    # resolves through the user-level kernelspec search path (audit T-8).
    client = nbclient.NotebookClient(nb, timeout=240, km=venv_kernel_manager)
    client.execute()

    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    assert n_code > 0
