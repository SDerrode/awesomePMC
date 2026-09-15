"""Smoke tests — verify the package imports and exposes its version."""

import re

import pmcprg


def test_version_is_defined():
    # Assert the *format* (PEP 440 major.minor.patch core), not a hardcoded
    # literal — the version lives in a single source (pmcprg/__init__.py, read by
    # Hatchling at build time) and must not need a lockstep edit here on every
    # release.
    assert isinstance(pmcprg.__version__, str)
    assert re.match(r"^\d+\.\d+\.\d+", pmcprg.__version__), pmcprg.__version__


def test_public_estimation_types_reexported():
    """The estimation result/trace types are part of the public surface and
    must be importable straight from ``pmcprg.pmc`` (audit A-4)."""
    from pmcprg.pmc import IceResult, IceTrace, SemResult, SemTrace

    # SEM types subclass the ICE ones (shared GUI/plotting code).
    assert issubclass(SemTrace, IceTrace)
    assert issubclass(SemResult, IceResult)
