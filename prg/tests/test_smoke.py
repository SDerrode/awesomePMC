"""Smoke tests — verify the package imports and exposes its version."""

import prg


def test_version_is_defined():
    assert isinstance(prg.__version__, str)
    assert prg.__version__ == "0.3.4"
