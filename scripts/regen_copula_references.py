#!/usr/bin/env python3
"""Regenerate or check the decimal references of ``pmcprg/tests/test_copula_limits.py``.

The copula edge tests compare the package with references computed in Python
``decimal`` at adaptive precision — minutes of work — which they read from
``pmcprg/tests/data/copula_limits_references.json``. This script recomputes
every row with the ``decimal`` code of the test module itself (nothing is
duplicated here) and writes the file, or checks it.

Usage, from the repository root::

    python scripts/regen_copula_references.py            # recompute and write the file
    python scripts/regen_copula_references.py --check    # recompute; exit 1 if the file differs
    python scripts/regen_copula_references.py -j 1       # serial (default: one process per CPU)

Regenerate after changing the decimal reference code, the cases (families, τ,
grid) or a τ → θ map of the package; review the diff before committing it.
The same full check runs inside pytest with ``PMC_REGEN_COPULA_REFS=1``.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "pmcprg" / "tests", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import test_copula_limits as tcl  # noqa: E402  (after the sys.path set-up)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="do not write; exit 1 if the recomputed references differ from the file")
    parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1,
                        help="worker processes (default: %(default)s)")
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            table = tcl._compute_reference_table(mapper=pool.map)
    else:
        table = tcl._compute_reference_table()
    elapsed = time.perf_counter() - t0

    rows = [row for key in table for row in table[key]]
    precisions = collections.Counter(row[5] for row in rows)
    not_converged = [(key, row[:2]) for key in table for row in table[key] if not row[6]]
    print(f"{len(table)} entries, {len(rows)} rows in {elapsed:.1f} s ({args.jobs} process(es)); "
          f"precision reached: {dict(sorted(precisions.items()))}")
    for key, uv in not_converged:
        print(f"NOT CONVERGED: {key} at (u, v) = {tuple(uv)}", file=sys.stderr)

    text = tcl._dump_reference_table(table)
    path = tcl._REF_FILE
    if args.check:
        old = path.read_text(encoding="utf-8") if path.exists() else None
        if old == text:
            print(f"OK: {path.relative_to(ROOT)} matches the decimal references")
            return 0
        diff = tcl._table_differences(tcl._parse_reference_table(old), table) if old else ["file missing"]
        print(f"DIFFERS: {path.relative_to(ROOT)} ({len(diff)} differences"
              f"{'; formatting only' if not diff else ''})", file=sys.stderr)
        for line in diff[:20]:
            print(f"  {line}", file=sys.stderr)
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
