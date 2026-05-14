"""
test_uci_har_notebook.py — execute the UCI HAR notebook end-to-end with the
synthetic fallback (no network).

The notebook lives at ``examples/uci_har_smartphone.ipynb``. We patch the
top cell to set ``ALLOW_DOWNLOAD = False`` so the regression test never
touches the network, then execute every code cell with a fresh kernel.

Tagged ``slow`` because it spins up a Jupyter kernel and runs ICE + SEM
+ MKS plus matplotlib plotting (~10–15 s).
"""

from __future__ import annotations

from pathlib import Path

import pytest


nbformat  = pytest.importorskip("nbformat")
nbclient  = pytest.importorskip("nbclient")
ipykernel = pytest.importorskip("ipykernel")   # noqa: F401  (kernel runtime)


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK  = REPO_ROOT / "examples" / "uci_har_smartphone.ipynb"


def _patch_allow_download_false(nb) -> None:
    """Find the cell setting ALLOW_DOWNLOAD and force it to False.

    Mutates ``nb.cells`` in place. Asserts that the cell exists so we
    catch any future notebook restructuring that would silently re-enable
    the network path in CI.
    """
    found = False
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if "ALLOW_DOWNLOAD" in cell.source:
            cell.source = cell.source.replace(
                "ALLOW_DOWNLOAD = True",
                "ALLOW_DOWNLOAD = False",
            )
            found = True
            break
    if not found:                                  # pragma: no cover
        raise AssertionError(
            "No code cell defining ALLOW_DOWNLOAD found in notebook — has "
            "the layout changed?"
        )


@pytest.mark.slow
def test_uci_har_notebook_executes_with_synthetic_fallback():
    """Every code cell of examples/uci_har_smartphone.ipynb must run with the
    synthetic fallback (no network)."""
    assert NOTEBOOK.exists(), f"missing notebook: {NOTEBOOK}"

    nb = nbformat.read(NOTEBOOK, as_version=4)
    _patch_allow_download_false(nb)

    client = nbclient.NotebookClient(nb, timeout=240, kernel_name="python3")
    client.execute(cwd=str(REPO_ROOT))

    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    assert n_code > 0
