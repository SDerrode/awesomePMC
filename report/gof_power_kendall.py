#!/usr/bin/env python3
"""Does the Kendall-process statistic turn separation into power? (audit S-6)

``report/gof_statistic_panel.py`` compared five ξ-weighted statistics on the
same datasets and found one clear winner: the Cramér-von Mises statistic on
the **Kendall process** — the law of C(U, V) rather than C itself. Paired
gain over the shipped ``cvm``, diagonal pairs: 1.465 [1.181, 1.826] against
GH and 2.795 [2.340, 3.442] against Clayton, where variance stabilisation
(``ad``) and a direct tail-dependence discrepancy (``tail``) both failed to
beat the baseline at all.

Separation is only a proxy. This script runs the same power study as
``report/gof_power_study.py`` — model-level parametric bootstrap, level
measured on its own H₀ configuration — with the statistic swapped, so the
question becomes the one that matters: does the heatmap start rejecting the
models it should?

The test replicated here mirrors ``PMCMainWindow._do_gof_test`` exactly
(pseudo-observations from the fitted margins, ξ weights, replicates
simulated from the fitted model), with one implementation note: K_θ depends
only on the fitted copula, not on the data, so it is computed **once per
pair** and reused across all B replicates. Recomputing it per replicate, as
the panel script does, would triple the cost for nothing.

Pseudo-observations of the pair (i, j) are ``(F_ij(y_n), F_ji(y_{n+1}))``
weighted by ``ξ_n(i, j)`` (:func:`pmcprg.diagnostics.copula_pseudo_obs`): the pair
margins of a general PMC (DerrodePieczynski_CSDA2013 Eqs. 12–14), which reduce
to the state margins ``(F_i(y_n), F_j(y_{n+1}))`` — the same floats — when
``f_ij = f_i``.

Usage
-----
    python report/gof_power_kendall.py --reps 100 --B 200 --workers 8
    python report/gof_power_kendall.py --report-only
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

OUT_DIR  = REPORT_DIR / "out" / "gof_power_kendall"
OUT_FILE = OUT_DIR / "results.jsonl"

BASE_MODEL = REPO_ROOT / "pmcprg" / "pmc" / "models" / "pmc_gauss_k2.toml"
FAMILIES   = ["gauss", "clayton", "gh"]          # gauss = H0
STATISTICS = ["cvm", "kendall"]
N_DEFAULT  = 1500

_KGRID = None            # lazily built; the Kendall-process evaluation grid


def _grid():
    global _KGRID
    if _KGRID is None:
        import numpy as np
        _KGRID = np.linspace(0.01, 0.99, 99)
    return _KGRID


# ---------------------------------------------------------------------------

def _build_models(family: str):
    from pmcprg.pmc.model import PMCModel

    base = PMCModel(BASE_MODEL)

    def _with(name: str):
        raw = base.raw
        for blk in raw["copulas"]:
            if abs(float(blk["tau"])) > 1e-9:
                blk["name"] = name
        return PMCModel.from_dict(raw)

    short = {"gauss": "Gauss", "clayton": "Clayton", "gh": "GH"}[family]
    return _with(short), _with("Gauss")


def _kendall_reference(cop, m_mc: int = 20_000, seed: int = 12345):
    """K_θ(t) = P(C_θ(U,V) ≤ t) on the grid — depends only on the copula.

    Computed once per pair and reused for every bootstrap replicate: it is a
    property of the fitted copula, not of the data.
    """
    import numpy as np

    sample = cop.sample(n=m_mc, seed=seed)
    W = cop.cdf_array(sample)
    return np.array([float(np.mean(W <= t)) for t in _grid()])


def _statistic(name, uv, w, cop, k_ref=None):
    import numpy as np

    from pmcprg.copulas._fit import _empirical_copula, _normalised_weights

    wn = _normalised_weights(w)
    if name == "cvm":
        Cn = _empirical_copula(uv, uv, w)
        return float(np.dot(wn, (Cn - cop.cdf_array(uv)) ** 2))
    if name == "kendall":
        Wn = _empirical_copula(uv, uv, w)
        Kn = np.array([float(np.sum(wn[Wn <= t])) for t in _grid()])
        return float(np.mean((Kn - k_ref) ** 2))
    raise ValueError(name)


def _pair_stats(mdl, Yv, stat, pairs, cops, sel, k_refs):
    from pmcprg.diagnostics   import copula_pseudo_obs, margin_cdfs
    from pmcprg.pmc.ice       import _effective_n
    from pmcprg.pmc.inference import (
        backward, forward, joint_posteriors, precompute_weights,
    )

    W, f_pdf = precompute_weights(mdl, Yv)
    alpha, _ = forward(mdl, Yv, W=W, f_pdf=f_pdf)
    xi       = joint_posteriors(alpha, W, backward(mdl, Yv, W=W))
    F        = margin_cdfs(mdl, Yv, clip=1e-12)          # F[n, i, j] = F_ij(y_n)

    out = {}
    for i, j in pairs:
        uv, w = copula_pseudo_obs(F, xi, i, j, sel)
        try:
            s = _statistic(stat, uv, w, cops[(i, j)], k_refs.get((i, j)))
        except Exception:
            s = float("nan")
        out[(i, j)] = (s, float(_effective_n(w)))
    return out


def run_one(task) -> dict:
    """One dataset: simulate under `family`, test the Gaussian model."""
    family, N, rep, B, stat = task

    import numpy as np

    from pmcprg.pmc.simulate import simulate

    gen, test = _build_models(family)
    _, Y = simulate(gen, N=N, seed=100_000 + 1000 * rep + N)

    pairs = [(int(b["i"]), int(b["j"])) for b in test.copula_blocks()]
    cops  = {p: test.copula(*p) for p in pairs}
    sel   = np.arange(0, len(Y) - 1)
    k_refs = ({p: _kendall_reference(cops[p]) for p in pairs}
              if stat == "kendall" else {})

    t0  = time.perf_counter()
    obs = _pair_stats(test, Y, stat, pairs, cops, sel, k_refs)
    null = {p: [] for p in pairs}
    rng = np.random.default_rng(rep)
    for _ in range(B):
        _, Yb = simulate(test, N=N, seed=int(rng.integers(0, 2**31 - 1)))
        for p, (s, _n) in _pair_stats(test, Yb, stat, pairs, cops,
                                      sel, k_refs).items():
            if np.isfinite(s):
                null[p].append(s)

    rows = []
    for p in pairs:
        s, n_eff = obs[p]
        draws = np.asarray(null[p], dtype=float)
        rows.append({
            "i": p[0], "j": p[1], "stat": float(s), "n_eff": n_eff,
            "p": (float(np.mean(draws >= s)) if draws.size and np.isfinite(s)
                  else float("nan")),
        })
    return {"family": family, "N": N, "rep": rep, "B": B, "statistic": stat,
            "seconds": round(time.perf_counter() - t0, 2), "pairs": rows}


# ---------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def summarise(rows) -> None:
    import numpy as np

    by: dict = {}
    for row in rows:
        for pr in row["pairs"]:
            if not np.isfinite(pr.get("p", float("nan"))):
                continue
            kind = "diag" if pr["i"] == pr["j"] else "off"
            by.setdefault((row["statistic"], row["family"], kind), []).append(pr["p"])

    print(f"\n{'statistique':<12}{'données':<10}{'paires':<7}{'n':>6}"
          f"{'rej 5%':>9}{'Wilson 95%':>18}{'rej 10%':>9}{'p méd':>8}")
    print("-" * 80)
    for (stat, family, kind), ps in sorted(by.items()):
        p = np.asarray(ps)
        k5 = int((p < 0.05).sum())
        lo, hi = wilson(k5, p.size)
        print(f"{stat:<12}{family:<10}{kind:<7}{p.size:>6}{k5 / p.size:>9.3f}"
              f"{f'[{lo:.3f}, {hi:.3f}]':>18}"
              f"{float((p < 0.10).mean()):>9.3f}{float(np.median(p)):>8.3f}")


def load(path: Path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--N", type=int, default=N_DEFAULT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--families", default=",".join(FAMILIES),
                    help="restrict the grid, e.g. `gauss` for a level-only run")
    ap.add_argument("--statistics", default=",".join(STATISTICS))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load(OUT_FILE)
    if args.report_only:
        print(f"{len(rows)} datasets in {OUT_FILE}")
        summarise(rows)
        return

    done = {(r["statistic"], r["family"], r["N"], r["rep"], r.get("B"))
            for r in rows}
    tasks = [
        (family, args.N, rep, args.B, stat)
        for rep in range(args.reps)
        for stat in args.statistics.split(",")
        for family in args.families.split(",")
        if (stat, family, args.N, rep, args.B) not in done
    ]
    if not tasks:
        print("Grid complete.")
        summarise(rows)
        return

    print(f"{len(tasks)} datasets, B={args.B}, N={args.N}, "
          f"{args.workers} workers -> {OUT_FILE}")
    t0 = time.perf_counter()
    with OUT_FILE.open("a") as fh, \
            ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, t): t for t in tasks}
        for n_done, fut in enumerate(as_completed(futures), 1):
            try:
                row = fut.result()
            except Exception as exc:
                print(f"  !! {futures[fut][:3]} : {type(exc).__name__}: {exc}",
                      flush=True)
                continue
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if n_done % 20 == 0 or n_done == len(tasks):
                el = time.perf_counter() - t0
                print(f"  {n_done}/{len(tasks)}  {el / 60:.1f} min  "
                      f"eta {el / n_done * (len(tasks) - n_done) / 60:.1f} min",
                      flush=True)
    summarise(load(OUT_FILE))


if __name__ == "__main__":            # required under the spawn start method
    main()
