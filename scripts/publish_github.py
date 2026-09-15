#!/usr/bin/env python3
"""Export the public tree of awesomePMC to a local ``public`` branch, for the public GitHub repository.

GitLab (``origin``) is the private working repository with the full history.
The public GitHub repository receives an *exported* history: each export is one
commit whose tree is the tree of a source commit (default ``HEAD``) minus the
private paths below, and whose parent is the previous export. Private files
therefore never appear in any public commit.

Usage, from the repository root::

    python scripts/publish_github.py                  # dry run: show what the export would contain
    python scripts/publish_github.py --apply          # create/advance refs/heads/public
    git push github public:main                       # then publish (explicitly, by hand)

Options: ``--source REV`` (default HEAD), ``--ref`` (default ``refs/heads/public``),
``--message`` (default: "Public export of <short sha> — <subject>").

The script never pushes. It refuses to export if the source tree still
contains a private path after filtering, and warns about text files in the
public tree that still mention the private paths.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Pathspecs removed from the public tree (git glob pathspecs, repository root).
PRIVATE_PATHSPECS = [
    ":(glob)AUDIT*.md",
    ":(glob)note/**",
]
PRIVATE_PATTERN = re.compile(r"^(AUDIT[^/]*\.md|note/)")
MENTION_PATTERN = re.compile(r"AUDIT[A-Z_]*\.md|(?<![A-Za-z0-9_/])note/")
TEXT_SUFFIXES = {".md", ".py", ".toml", ".cff", ".tex", ".yml", ".yaml", ".txt", ".cfg", ".ipynb", ""}


def git(*args: str, env: dict | None = None, input_: str | None = None) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True,
        env={**os.environ, **(env or {})}, input=input_,
    ).stdout.strip()


def ref_exists(ref: str) -> bool:
    return subprocess.run(["git", "rev-parse", "--verify", "--quiet", ref],
                          cwd=ROOT, capture_output=True).returncode == 0


def public_tree(source: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        env = {"GIT_INDEX_FILE": str(Path(tmp) / "index")}
        git("read-tree", source, env=env)
        git("rm", "--cached", "-r", "--quiet", "--ignore-unmatch", "--", *PRIVATE_PATHSPECS, env=env)
        return git("write-tree", env=env)


def tree_files(tree: str) -> list[str]:
    return [p for p in git("ls-tree", "-r", "--name-only", "-z", tree).split("\0") if p]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default="HEAD", help="commit to export (default HEAD)")
    ap.add_argument("--ref", default="refs/heads/public", help="local ref holding the public history")
    ap.add_argument("--message", default=None, help="commit message of the export")
    ap.add_argument("--apply", action="store_true", help="write the export commit and move --ref")
    args = ap.parse_args()

    source = git("rev-parse", "--verify", f"{args.source}^{{commit}}")
    tree = public_tree(source)
    files = tree_files(tree)
    leaked = [f for f in files if PRIVATE_PATTERN.match(f)]
    if leaked:
        print("refusing: private paths remain in the public tree:", *leaked, sep="\n  ", file=sys.stderr)
        return 2

    removed = sorted(set(tree_files(source)) - set(files))
    mentions = []
    for f in files:
        if Path(f).suffix.lower() not in TEXT_SUFFIXES:
            continue
        blob = git("show", f"{tree}:{f}")
        n = len(MENTION_PATTERN.findall(blob))
        if n:
            mentions.append((f, n))

    parent = git("rev-parse", args.ref) if ref_exists(args.ref) else None
    unchanged = parent is not None and git("rev-parse", f"{parent}^{{tree}}") == tree
    short, subject = git("log", "-1", "--format=%h%x00%s", source).split("\0")

    print(f"source   {source[:12]}  {subject}")
    print(f"public   tree {tree[:12]}  {len(files)} files  ({len(removed)} private files removed)")
    for f in removed:
        print(f"  - {f}")
    if mentions:
        print("warning: public files still mention private paths:")
        for f, n in mentions:
            print(f"  {f}: {n}")
    print(f"ref      {args.ref} -> {parent[:12] if parent else '(new, root commit)'}")
    if unchanged:
        print("nothing to export: the public tree is identical to the last export")
        return 0
    if not args.apply:
        print("dry run — nothing written (use --apply)")
        return 0

    message = args.message or f"Public export of {short} — {subject}"
    cmd = ["commit-tree", tree, "-m", message]
    if parent:
        cmd[2:2] = ["-p", parent]
    commit = git(*cmd)
    git("update-ref", args.ref, commit, *( [parent] if parent else [] ))
    print(f"exported {commit[:12]} on {args.ref}")
    print(f"to publish: git push github {args.ref.removeprefix('refs/heads/')}:main")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
