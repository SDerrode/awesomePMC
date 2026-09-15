#!/usr/bin/env python3
"""Rename the project ``copulasformm`` → ``awesomePMC`` and its import package ``prg`` → ``pmcprg``.

Run from the repository root on a clean working tree::

    python scripts/rename_to_awesomepmc.py            # dry run: list what would change
    python scripts/rename_to_awesomepmc.py --apply    # git mv prg pmcprg + rewrite files

Rules
-----
* The directory ``prg/`` is moved with ``git mv`` to ``pmcprg/`` (history kept).
* In every tracked text file, the token ``prg`` — not preceded or followed by a
  letter, digit or underscore — becomes ``pmcprg``: imports (``from prg.pmc``),
  qualified names in strings (``"prg.pmc.ice"``, logger ``"prg"``), paths
  (``prg/pmc/models``), ``pyproject`` entries (``packages = ["prg"]``).
* Project name: the PyPI distribution becomes ``awesomepmc`` (``pip install
  'awesomepmc[ml]'``, ``name = "awesomepmc"``), the per-user directory
  ``~/.copulasformm`` becomes ``~/.awesomepmc``, the repository URLs
  ``…/sderrode/copulasformm`` and ``…/SDerrode/copulasformm`` end in
  ``awesomePMC``, and every other mention reads ``awesomePMC``.
* Not rewritten (history): ``CHANGELOG.md`` entries, ``AUDIT*.md`` reports,
  generated logs under ``report/results/``, binary files, and this script.
  A CHANGELOG entry describing the rename is inserted under ``[Unreleased]``.

The script refuses to run on a dirty tree or if ``pmcprg/`` already exists.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()

PRG_TOKEN = re.compile(r"(?<![A-Za-z0-9_])prg(?![A-Za-z0-9_])")
NAME_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r'name = "copulasformm"'), 'name = "awesomepmc"'),
    (re.compile(r"copulasformm\["), "awesomepmc["),
    (re.compile(r"\.copulasformm\b"), ".awesomepmc"),
    (re.compile(r"(gitlab\.ec-lyon\.fr/sderrode/)copulasformm"), r"\1awesomePMC"),
    (re.compile(r"(github\.com/SDerrode/)copulasformm"), r"\1awesomePMC"),
    (re.compile(r"copulasformm"), "awesomePMC"),
]
SKIP_EXACT = {"CHANGELOG.md"}
SKIP_PATTERNS = [re.compile(r"^AUDIT[^/]*\.md$"), re.compile(r"^report/results/.*\.log$")]
BINARY_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".npy", ".npz", ".pkl", ".zip", ".gz", ".ico", ".icns"}

CHANGELOG_ENTRY = """### Changed — project renamed `awesomePMC`, import package `pmcprg`

- **The project is now `awesomePMC`** (PyPI distribution `awesomepmc`) and the
  import package `prg` is now **`pmcprg`**: `from pmcprg.pmc import ice`,
  `python -m pmcprg.pmc`, `pip install 'awesomepmc[ml]'`. The command-line
  entry point is still `pmc`. `prg` clashed with other packages installing a
  top-level `prg` (e.g. awesomePKF) in the same environment.
- The per-user log directory moves from `~/.copulasformm` to `~/.awesomepmc`;
  the GUI settings are stored under the organisation `awesomePMC` (previous
  settings are not migrated).
- Repository URLs: `gitlab.ec-lyon.fr/sderrode/awesomePMC` and the GitHub mirror
  `SDerrode/awesomePMC` (the old URLs redirect).
- File paths `prg/…` quoted in earlier CHANGELOG entries and in the `AUDIT*.md`
  reports refer to the pre-rename layout (`pmcprg/…` now).

"""


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def tracked_files() -> list[str]:
    return [f for f in git("ls-files", "-z").split("\0") if f]


def skipped(rel: str) -> bool:
    if rel in SKIP_EXACT or (ROOT / rel).resolve() == SELF:
        return True
    if any(p.match(rel) for p in SKIP_PATTERNS):
        return True
    return Path(rel).suffix.lower() in BINARY_SUFFIXES


def rewrite(text: str, rel: str) -> str:
    out = PRG_TOKEN.sub("pmcprg", text)
    if rel == ".gitignore":
        out = out.replace("copulasformm.code-workspace", "copulasformm.code-workspace\nawesomePMC.code-workspace")
        return out
    for pat, repl in NAME_RULES:
        out = pat.sub(repl, out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true", help="perform the rename (default: dry run)")
    args = ap.parse_args()

    if git("status", "--porcelain").strip():
        print("refusing: the working tree is not clean", file=sys.stderr)
        return 2
    if (ROOT / "pmcprg").exists():
        print("refusing: pmcprg/ already exists", file=sys.stderr)
        return 2
    if not (ROOT / "prg").is_dir():
        print("refusing: prg/ not found", file=sys.stderr)
        return 2

    changes: list[tuple[str, int, int]] = []
    for rel in tracked_files():
        if skipped(rel):
            continue
        path = ROOT / rel
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        if "\0" in text:
            continue
        new = rewrite(text, rel)
        if new != text:
            changes.append((rel, len(PRG_TOKEN.findall(text)), text.count("copulasformm")))
            if args.apply:
                path.write_text(new, encoding="utf-8")

    n_prg = sum(c[1] for c in changes)
    n_name = sum(c[2] for c in changes)
    for rel, a, b in changes:
        print(f"{rel}: prg×{a} copulasformm×{b}")
    print(f"\n{len(changes)} files, {n_prg} 'prg' tokens, {n_name} 'copulasformm' mentions"
          + ("" if args.apply else "  (dry run — nothing written)"))

    if args.apply:
        changelog = ROOT / "CHANGELOG.md"
        text = changelog.read_text(encoding="utf-8")
        anchor = "## [Unreleased]\n\n"
        if anchor in text and "project renamed `awesomePMC`" not in text:
            changelog.write_text(text.replace(anchor, anchor + CHANGELOG_ENTRY, 1), encoding="utf-8")
        git("mv", "prg", "pmcprg")
        print("git mv prg pmcprg — done. Next: run the test suite, then commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
