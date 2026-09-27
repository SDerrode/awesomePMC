"""
robust_helper.py — a second process pool for a running ``robust.py --resume``
(erroneous data).

``robust.py`` saves every task to ``results/cache/robust_tasks/`` as it
finishes and, with ``--resume``, loads a saved task instead of running it
when its turn comes. This helper runs the same task specifications in the
REVERSE of robust.py's order, into the same cache: the two runs meet in the
middle, and each of the helper's workers can at most duplicate the task the
main run is on. The fits are deterministic, so a duplicate saves the same
result (the rerun of 2026-09-26: the same iterations and masks per fit in
both logs).

Start it after ``robust.py --resume`` with the same ``--max-rounds``. Stop
it once only the tasks the main run is on are left: stop its WORKERS too
(``pkill -P <helper pid>`` then the helper; a pool worker outlives its
parent). Once both have stopped, a last ``robust.py --resume`` loads every
task and writes the tables.

Usage (from the repository root)::

    .venv/bin/python report/erroneous_data/intel_lab/robust_helper.py --jobs 4
    .venv/bin/python report/erroneous_data/intel_lab/robust_helper.py --jobs 4 --kinds pmc_state
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import il_common as C
import robust as R


def specs(args) -> list[dict]:
    """robust.run's task specifications (the same cache keys), in the reverse of its order."""
    sel = C.selected_models()
    out = [{"mote": m, "kind": k, "K": sel[(m, k)][0], "method": meth, "data": args.data,
            "max_rounds": args.max_rounds, "resume": True}
           for m in C.MOTES for k in C.KINDS for meth in R.METHODS]
    out.sort(key=lambda s: (s["kind"] != "pmc_state", not s["method"].startswith("fm")))
    if args.kinds:
        out = [s for s in out if s["kind"] in args.kinds]
    return out[::-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data", default=C.DEFAULT_DATA)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--max-rounds", type=int, default=R.MAX_ROUNDS,
                    help="must equal the main run's (it is part of the cache key)")
    ap.add_argument("--kinds", nargs="+", choices=C.KINDS,
                    help="only these model kinds (default: all)")
    args = ap.parse_args()
    print("pmcprg:", C.check_import())
    todo = specs(args)
    print(f"{len(todo)} task(s), in reverse order, first: "
          + ", ".join(f"{s['mote']} {s['kind']} {s['method']}" for s in todo[:3]), flush=True)
    with ProcessPoolExecutor(args.jobs) as ex:
        list(ex.map(R.task, todo))
    print("helper done", flush=True)


if __name__ == "__main__":
    main()
