#!/usr/bin/env python3
"""Level and power of the ξ-weighted GoF test (audit S-6).

Why this exists
---------------
The S-6 fix made the GoF heatmap a genuine per-pair diagnostic, and its
*level* was checked on the spot (40 datasets, B = 100, N = 400: the diagonal
rejected at 0.025 and the off-diagonal at 0.062 for a nominal 0.05). Its
*power* was not: three quick estimates at N = 400 / 1500 / 4000 gave
rejection rates of 0.087, 0.150 and 0.067 whose Wilson intervals overlap
completely and all contain the nominal 0.05. Nothing could be concluded —
neither that the test has power, nor that it lacks it. That is a limit of
the protocol (R = 15…40 datasets, B = 40…60 replicates) as much as of the
test.

This script runs the study properly: R datasets per configuration, B
bootstrap replicates each, so the Wilson interval on a rejection rate is
about ±0.04 rather than ±0.13.

Design
------
Data are generated from a PMC whose *non-independent* pair copulas are
replaced by one family, and tested against a model identical except that its
copulas are declared Gaussian, at the **same Kendall τ**. Matching τ is the
point: the alternative differs from the null only in tail behaviour, which is
the discrimination the CvM statistic is supposed to provide.

  * ``gauss``   — H₀. Gives the empirical level, and the reference for the
                  H1/H0 statistic ratio.
  * ``clayton`` — lower-tail dependence, no upper.
  * ``gh``      — upper-tail dependence, no lower (Gumbel-Hougaard).

Diagonal pairs (i = j) carry the dependence in this fixture; the off-diagonal
ones are independent by construction (τ = 0), so their rejection rate is a
second, independent read on the level and is reported separately.

Cost and interruption
---------------------
Measured at ~59 ms per bootstrap replicate at N = 400 and ~234 ms at
N = 1500, i.e. roughly 1.6 and 6.5 CPU-hours per configuration at
R = 200 / B = 500. Tasks are emitted **replicate-major**, so every
configuration advances together and the study is balanced at any moment: kill
it whenever and the partial JSONL is still a valid, if smaller, study.
Re-running skips whatever is already recorded.

Usage
-----
    python report/gof_power_study.py                     # full grid
    python report/gof_power_study.py --reps 40 --B 200   # a quick look
    python report/gof_power_study.py --report-only       # just summarise
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPORT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = REPORT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_DIR  = REPORT_DIR / "out" / "gof_power"
OUT_FILE = OUT_DIR / "results.jsonl"

BASE_MODEL = REPO_ROOT / "pmcprg" / "pmc" / "models" / "pmc_gauss_k2.toml"

FAMILIES = ["gauss", "clayton", "gh"]        # gauss = H0
SIZES    = [400, 1500]


# ---------------------------------------------------------------------------
# Worker — must be importable at module level for the spawn start method.
# ---------------------------------------------------------------------------

def _build_models(family: str):
    """(generating model, model under test) — same τ, different family.

    The model under test is always Gaussian: the question is whether the GoF
    test notices that the data were not.
    """
    from pmcprg.pmc.model import PMCModel

    base = PMCModel(BASE_MODEL)

    def _with(name: str):
        raw = base.raw
        for blk in raw["copulas"]:
            if abs(float(blk["tau"])) > 1e-9:      # leave independent pairs alone
                blk["name"] = name
        return PMCModel.from_dict(raw)

    short = {"gauss": "Gauss", "clayton": "Clayton", "gh": "GH"}[family]
    return _with(short), _with("Gauss")


def run_one(task: tuple[str, int, int, int]) -> dict:
    """One dataset: simulate under `family`, test against the Gaussian model."""
    family, N, rep, B = task

    from PyQt6.QtWidgets import QApplication

    if QApplication.instance() is None:
        QApplication([])                          # never shown; offscreen

    from pmcprg.pmc.gui.main_window import PMCMainWindow
    from pmcprg.pmc.simulate        import simulate

    gen, test = _build_models(family)
    _, Y = simulate(gen, N=N, seed=100_000 + 1000 * rep + N)

    t0 = time.perf_counter()
    results = PMCMainWindow._do_gof_test(
        test, Y, B=B, seed=rep, max_points=N, max_len=N,
    )
    return {
        "family": family, "N": N, "rep": rep, "B": B,
        "seconds": round(time.perf_counter() - t0, 2),
        "pairs": [
            {k: r[k] for k in ("i", "j", "p", "stat", "n_eff") if k in r}
            for r in results
        ],
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — the honest one for proportions near 0."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half   = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def summarise(rows: list[dict]) -> None:
    import numpy as np

    by: dict[tuple, dict[str, list]] = {}
    for row in rows:
        for pair in row["pairs"]:
            if "p" not in pair:
                continue
            kind = "diag" if pair["i"] == pair["j"] else "off"
            slot = by.setdefault((row["family"], row["N"], kind),
                                 {"p": [], "stat": [], "n_eff": []})
            slot["p"].append(pair["p"])
            slot["stat"].append(pair["stat"])
            slot["n_eff"].append(pair["n_eff"])

    print(f"\n{'config':<26}{'n':>6}{'rej 5%':>9}{'Wilson 95%':>18}"
          f"{'rej 10%':>9}{'p med':>8}{'n_eff':>8}")
    print("-" * 84)
    for (family, N, kind), slot in sorted(by.items()):
        p = np.asarray(slot["p"])
        k5 = int((p < 0.05).sum())
        lo, hi = wilson(k5, p.size)
        label = f"{family} N={N} {kind}"
        print(f"{label:<26}{p.size:>6}{k5 / p.size:>9.3f}"
              f"{f'[{lo:.3f}, {hi:.3f}]':>18}"
              f"{float((p < 0.10).mean()):>9.3f}{float(np.median(p)):>8.3f}"
              f"{float(np.median(slot['n_eff'])):>8.0f}")

    # H1/H0 separation of the statistic itself, diagonal pairs only.
    print("\nSéparation de la statistique (paires diagonales, médianes) :")
    for N in sorted({N for _, N, _ in by}):
        h0 = by.get(("gauss", N, "diag"))
        if not h0:
            continue
        s0 = float(np.median(h0["stat"]))
        for family in FAMILIES:
            if family == "gauss":
                continue
            h1 = by.get((family, N, "diag"))
            if not h1:
                continue
            s1 = float(np.median(h1["stat"]))
            print(f"  N={N:<6} {family:<9} H0={s0:.3e}  H1={s1:.3e}  "
                  f"rapport={s1 / max(s0, 1e-300):.2f}")


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:          # a torn last line after a kill
                pass
    return out


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done_rows = load(OUT_FILE)
    if args.report_only:
        print(f"{len(done_rows)} datasets in {OUT_FILE}")
        summarise(done_rows)
        return

    # B is part of the key: a run at a different B is a different study, and
    # silently inheriting rows from a cheaper smoke run would poison it.
    done = {(r["family"], r["N"], r["rep"], r.get("B")) for r in done_rows}
    done_rows = [r for r in done_rows if r.get("B") == args.B]

    # Replicate-major: every configuration advances together, so the study is
    # balanced at any moment and killing it early still leaves a valid one.
    tasks = [
        (family, N, rep, args.B)
        for rep in range(args.reps)
        for N in SIZES
        for family in FAMILIES
        if (family, N, rep, args.B) not in done
    ]
    if not tasks:
        print("Nothing to do — the grid is complete.")
        summarise(done_rows)
        return

    print(f"{len(tasks)} datasets to run "
          f"({len(done)} already recorded), B={args.B}, "
          f"{args.workers} workers -> {OUT_FILE}")

    t0 = time.perf_counter()
    with OUT_FILE.open("a") as fh, \
            ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, t): t for t in tasks}
        for n_done, fut in enumerate(as_completed(futures), 1):
            task = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:               # keep the study going
                print(f"  !! {task[:3]} failed: {type(exc).__name__}: {exc}",
                      flush=True)
                continue
            # Flush per dataset: a kill must never cost more than one.
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if n_done % 10 == 0 or n_done == len(tasks):
                el  = time.perf_counter() - t0
                eta = el / n_done * (len(tasks) - n_done)
                print(f"  {n_done}/{len(tasks)}  elapsed {el / 60:.1f} min  "
                      f"eta {eta / 60:.1f} min", flush=True)

    summarise([r for r in load(OUT_FILE) if r.get("B") == args.B])


if __name__ == "__main__":          # required: spawn re-imports this module
    main()
