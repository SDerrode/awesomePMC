"""Guard the documented copula inventory against the registry.

``README.md`` is the PyPI front page, and it drifted: it advertised "17 copula
families" in three places and its table stopped at Plackett, long after the
registry had grown past both. Nothing failed, because no test ever compared
the prose to :class:`CopulaEnum`. These two do — cheaply, so they stay in the
fast suite.

Deliberately *not* a README generator: the repo had one and it was removed as
obsolete. The point here is only to make the drift loud, not to own the text.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pmcprg.copulas import CopulaEnum

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"
PACKAGE_INIT = REPO_ROOT / "pmcprg" / "__init__.py"

#: "39 copula families", "39 bivariate copula families", "**39 families**".
_COUNT_PATTERNS = (
    re.compile(r"(\d+)\s+(?:bivariate\s+)?copula families"),
    re.compile(r"\*\*(\d+) families\*\*"),
)

#: Every `Backticked` token, used to look up SHORT_NAMEs in the prose.
_BACKTICKED = re.compile(r"`([^`\n]+)`")


def _documents() -> list[Path]:
    return [p for p in (README, PACKAGE_INIT) if p.is_file()]


@pytest.mark.parametrize("path", _documents(), ids=lambda p: p.name)
def test_documented_family_count_matches_registry(path: Path) -> None:
    """Every "N copula families" claim in the docs equals the registry size."""
    expected = len(CopulaEnum.available())
    text = path.read_text(encoding="utf-8")

    found = [int(m) for pat in _COUNT_PATTERNS for m in pat.findall(text)]
    assert found, (
        f"{path.name} no longer states a copula-family count. If the wording "
        f"changed on purpose, update _COUNT_PATTERNS in "
        f"{Path(__file__).name}; otherwise restore the count (it is "
        f"{expected})."
    )

    wrong = sorted({n for n in found if n != expected})
    assert not wrong, (
        f"{path.name} advertises {wrong} copula families, but "
        f"CopulaEnum.available() holds {expected}. Update every occurrence of "
        f"the count in {path.name} to {expected} — in README.md that is the "
        f"'Why awesomePMC?' feature bullet, the sub-package table row for "
        f"`pmcprg.copulas`, the 'Available families' intro and the "
        f"'Folder structure' block — and add the new families to the tables "
        f"under 'Available families'."
    )


def test_readme_lists_every_registered_family() -> None:
    """Every SHORT_NAME in the registry appears in the README tables."""
    if not README.is_file():
        pytest.skip("README.md is not shipped in an installed distribution")

    documented = set(_BACKTICKED.findall(README.read_text(encoding="utf-8")))
    missing = [
        e.value.SHORT_NAME
        for e in CopulaEnum.available()
        if e.value.SHORT_NAME not in documented
    ]
    assert not missing, (
        f"README.md does not mention {len(missing)} registered copula "
        f"family/families: {', '.join(missing)}. Add a row for each under the "
        f"matching '#### …' heading of the 'Available families' section, with "
        f"its SHORT_NAME in backticks, its τ range and its tail dependence."
    )
