"""Measurements behind the design of the mechanism M-step of ICE / SEM (P6).

Three questions, each on HMC-IN (K = 2, A = [[0.95, 0.05], [0.05, 0.95]],
margins N(∓1, 1)), ICE with ``fit_margins = True``, ``tol = 1e-8``,
``max_iter = 500``, ``patience = 500`` (run to ICE's fixed point: the SR
symmetrisation of the prior makes ICE regress by ~1e-4 nat near it, which the
default ``patience = 3`` takes for a stop), ``missing_strategy = "available"``:

1. ``initial`` — the initial term γ_0(i) log p(m_0 | s_i) of "state-markov":
   the exact per-state maximisation against the closed form that drops it,
   both inside full ICE runs (masks a = (0.005, 0.03), b = (0.7, 0.9)).
2. ``guard`` — the pseudo-count c of the boundary guard: bias and RMSE of π̂
   ("state") and â, b̂ ("state-markov") for c ∈ {0, 0.1, 1, 10}, including a
   state that is never missing and a state with almost no bursts.
3. ``trap`` — a start with a rate at 0 (absorbing without the guard), and
   the common-rate start on data with and without a real state dependence
   (i.i.d. states, where π is not identified).

Seeds: ``zlib.crc32`` of named tuples; the path and the mask have distinct
seeds. Writes ``results/design_<part>.csv`` and prints the tables.

    .venv/bin/python report/missing_state/design_measurements.py --jobs 6
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = Path(__file__).resolve().parent / "results"
ICE_CFG = {"fit_margins": True, "tol": 1e-8, "max_iter": 500, "patience": 500}


def _seed(*parts) -> int:
    return zlib.crc32(repr(parts).encode()) % 2**31


def _hmc_in(A=((0.95, 0.05), (0.05, 0.95)), mu=(-1.0, 1.0)):
    from pmcprg.pmc import PMCModel
    return PMCModel.from_dict({
        "model": {"variant": "HMC-IN", "K": 2}, "prior": {"A": [list(r) for r in A]},
        "margins": [{"i": i, "dist": "norm", "params": {"loc": float(m), "scale": 1.0}}
                    for i, m in enumerate(mu)]})


def _data(model, N, tag, r, mech):
    from pmcprg.missing.patterns import state_dependent, state_markov
    from pmcprg.pmc import simulate
    X, Y = simulate(model, N=N, seed=_seed(tag, "x", N, r))
    if mech[0] == "state":
        Yn, mk = state_dependent(Y, X, mech[1], seed=_seed(tag, "mask", N, r))
    else:
        Yn, mk = state_markov(Y, X, mech[1], mech[2], seed=_seed(tag, "mask", N, r))
    return X, Yn, mk


def _fit(model, Yn, mode, *, c=None, closed_form=False, start=None, max_iter=None):
    import pmcprg.pmc.missingness as MS
    from pmcprg.pmc.ice import ice
    saved_c, saved_max = MS.PSEUDO_COUNT, MS._markov_maximise
    try:
        if c is not None:
            MS.PSEUDO_COUNT = float(c)
        if closed_form:
            MS._markov_maximise = lambda na1, na0, nb1, nb0, g0, start: start
        cfg = {**ICE_CFG, "missingness": mode}
        if max_iter is not None:
            cfg["max_iter"] = max_iter
        return ice(start if start is not None else model, Yn, cfg)
    finally:
        MS.PSEUDO_COUNT, MS._markov_maximise = saved_c, saved_max


# ---------------------------------------------------------------------------
# 1. initial term
# ---------------------------------------------------------------------------

A_TRUE, B_TRUE = (0.005, 0.03), (0.7, 0.9)


def _job_initial(args):
    N, r = args
    m = _hmc_in()
    _, Yn, mk = _data(m, N, "initial", r, ("markov", A_TRUE, B_TRUE))
    f1, t1 = _fit(m, Yn, "state-markov")
    f0, t0 = _fit(m, Yn, "state-markov", closed_form=True)
    return {"N": N, "r": r, "M": int(mk.sum()),
            **{f"a{i}": f1.missingness.onset[i] for i in range(2)},
            **{f"b{i}": f1.missingness.persistence[i] for i in range(2)},
            **{f"a{i}_cf": f0.missingness.onset[i] for i in range(2)},
            **{f"b{i}_cf": f0.missingness.persistence[i] for i in range(2)},
            "ll": t1.log_liks[-1], "ll_cf": t0.log_liks[-1],
            "iters": len(t1.log_liks), "iters_cf": len(t0.log_liks)}


def _summary_initial(rows):
    print("\n1. initial term of 'state-markov': exact vs closed form (ICE runs)")
    print(f"   truth a = {A_TRUE}, b = {B_TRUE}")
    print("   N     R   param  sd(exact)  mean|Δ|    max|Δ|     mean Δ/sd  | ΔLL: mean   max")
    for N in sorted({r["N"] for r in rows}):
        rs = [r for r in rows if r["N"] == N]
        dll = np.array([r["ll"] - r["ll_cf"] for r in rs])
        for k in ("a0", "a1", "b0", "b1"):
            x = np.array([r[k] for r in rs])
            d = x - np.array([r[k + "_cf"] for r in rs])
            print(f"   {N:<5d} {len(rs):<3d} {k:5s}  {x.std(ddof=1):.3e}  {np.abs(d).mean():.3e}  "
                  f"{np.abs(d).max():.3e}  {d.mean() / x.std(ddof=1):+.4f}"
                  + (f"   | {dll.mean():.2e}  {dll.max():.2e}" if k == "a0" else ""))


# ---------------------------------------------------------------------------
# 2. guard
# ---------------------------------------------------------------------------

GUARD_CASES = {
    "state-(0.02,0.3)": ("state", (0.02, 0.3)),
    "state-(0,0.2)":    ("state", (0.0, 0.2)),
    "markov-few-bursts": ("markov", (0.002, 0.03), (0.5, 0.9)),
}
CS = (0.0, 0.1, 1.0, 10.0)


def _job_guard(args):
    case, N, r = args
    m = _hmc_in()
    mech = GUARD_CASES[case]
    _, Yn, mk = _data(m, N, "guard-" + case, r, mech)
    mode = "state" if mech[0] == "state" else "state-markov"
    out = {"case": case, "N": N, "r": r, "M": int(mk.sum())}
    for c in CS:
        f, _ = _fit(m, Yn, mode, c=c)
        t = f.missingness.to_table()
        for k in ("rates", "onset", "persistence"):
            for i, v in enumerate(t.get(k, [])):
                out[f"{k}{i}_c{c:g}"] = v
    return out


def _summary_guard(rows):
    print("\n2. guard: bias and RMSE of the estimates against the pseudo-count c")
    for case, mech in GUARD_CASES.items():
        truth = ({"rates": mech[1]} if mech[0] == "state"
                 else {"onset": mech[1], "persistence": mech[2]})
        for N in sorted({r["N"] for r in rows if r["case"] == case}):
            rs = [r for r in rows if r["case"] == case and r["N"] == N]
            print(f"   {case}  N = {N}  R = {len(rs)}  (mean M = {np.mean([r['M'] for r in rs]):.0f})")
            for k, tv in truth.items():
                for i, t in enumerate(tv):
                    line = f"     {k}[{i}] = {t:<6g}"
                    for c in CS:
                        x = np.array([r[f"{k}{i}_c{c:g}"] for r in rs])
                        line += (f" | c={c:<4g} bias {x.mean() - t:+.4f} rmse "
                                 f"{math.sqrt(((x - t) ** 2).mean()):.4f} min {x.min():.1e}")
                    print(line)


# ---------------------------------------------------------------------------
# 3. traps
# ---------------------------------------------------------------------------

def _job_trap(args):
    kind, N, r = args
    from pmcprg.pmc import StateMissingness
    if kind == "zero-start":
        m = _hmc_in()
        _, Yn, mk = _data(m, N, "trap-zero", r, ("state", (0.05, 0.3)))
        start = m.with_missingness(StateMissingness(rates=(0.0, 0.3)))
        out = {"kind": kind, "N": N, "r": r, "M": int(mk.sum())}
        for c in (0.0, 1.0):
            f, t = _fit(m, Yn, "state", c=c, start=start)
            hist = np.array([h["rates"][0] for h in t.missingness_history])
            final = f.missingness.rates[0]
            near = np.nonzero(np.abs(hist - final) <= 0.1 * max(final, 1e-12))[0]
            out[f"pi0_c{c:g}"] = final
            out[f"iters_c{c:g}"] = len(t.log_liks)
            out[f"reach_c{c:g}"] = int(near[0]) if near.size else -1
            out[f"ll_c{c:g}"] = t.log_liks[-1]
        f, t = _fit(m, Yn, "state")                        # common start, for reference
        out["pi0_common"], out["ll_common"] = f.missingness.rates[0], t.log_liks[-1]
        return out
    # common start: state-dependent data vs i.i.d. states (π not identified)
    A = ((0.95, 0.05), (0.05, 0.95)) if kind == "common-markov-states" else ((0.5, 0.5), (0.5, 0.5))
    m = _hmc_in(A=A)
    _, Yn, mk = _data(m, N, "trap-" + kind, r, ("state", (0.02, 0.3)))
    f, t = _fit(m, Yn, "state")
    return {"kind": kind, "N": N, "r": r, "M": int(mk.sum()),
            "pi0_first": t.missingness_history[1]["rates"][0] if len(t.log_liks) > 1 else np.nan,
            "pi1_first": t.missingness_history[1]["rates"][1] if len(t.log_liks) > 1 else np.nan,
            "pi0": f.missingness.rates[0], "pi1": f.missingness.rates[1],
            "common": t.missingness_history[0]["rates"][0], "iters": len(t.log_liks)}


def _summary_trap(rows):
    print("\n3a. start with π_0 = 0 (truth π = (0.05, 0.3)); 'reach' = first iterate within 10 % of the final value")
    rs = [r for r in rows if r["kind"] == "zero-start"]
    for N in sorted({r["N"] for r in rs}):
        s = [r for r in rs if r["N"] == N]
        for c in (0.0, 1.0):
            pi0 = np.array([r[f"pi0_c{c:g}"] for r in s])
            reach = np.array([r[f"reach_c{c:g}"] for r in s])
            dll = np.array([r["ll_common"] - r[f"ll_c{c:g}"] for r in s])
            print(f"   N = {N}  c = {c:g}: π̂_0 mean {pi0.mean():.4f} (min {pi0.min():.2e}, "
                  f"exactly 0 in {int((pi0 == 0).sum())}/{len(s)}), reach median "
                  f"{np.median(reach):.0f} max {reach.max()}, "
                  f"LL(common start) − LL: mean {dll.mean():.3f} max {dll.max():.3f}")
    print("\n3b. common-rate start (truth π = (0.02, 0.3))")
    for kind in ("common-markov-states", "common-iid-states"):
        s = [r for r in rows if r["kind"] == kind]
        for N in sorted({r["N"] for r in s}):
            t = [r for r in s if r["N"] == N]
            p0 = np.array([r["pi0"] for r in t])
            p1 = np.array([r["pi1"] for r in t])
            f0 = np.array([r["pi0_first"] for r in t])
            f1 = np.array([r["pi1_first"] for r in t])
            cm = np.array([r["common"] for r in t])
            print(f"   {kind}  N = {N}  R = {len(t)}: common {cm.mean():.4f}; after 1 M-step "
                  f"π̂ = ({f0.mean():.4f}, {f1.mean():.4f}); final π̂ = ({p0.mean():.4f} ± "
                  f"{p0.std(ddof=1):.4f}, {p1.mean():.4f} ± {p1.std(ddof=1):.4f})")


def _quiet():
    logging.disable(logging.WARNING)


def _run(jobs, fn, args, workers):
    with ProcessPoolExecutor(max_workers=workers, initializer=_quiet) as pool:
        return list(pool.map(fn, args, chunksize=1))


def _write(rows, name):
    OUT.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / f"design_{name}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--parts", default="initial,guard,trap")
    args = ap.parse_args()
    logging.disable(logging.WARNING)
    R = args.reps
    parts = args.parts.split(",")
    t0 = time.time()
    if "initial" in parts:
        rows = _run(None, _job_initial, [(N, r) for N in (500, 2000) for r in range(R)], args.jobs)
        _write(rows, "initial")
        _summary_initial(rows)
    if "guard" in parts:
        rows = _run(None, _job_guard, [(c, N, r) for c in GUARD_CASES for N in (500, 2000)
                                        for r in range(R)], args.jobs)
        _write(rows, "guard")
        _summary_guard(rows)
    if "trap" in parts:
        jobs = ([("zero-start", N, r) for N in (500, 2000) for r in range(R // 2)]
                + [(k, N, r) for k in ("common-markov-states", "common-iid-states")
                   for N in (500, 2000) for r in range(R // 2)])
        rows = _run(None, _job_trap, jobs, args.jobs)
        _write(rows, "trap")
        _summary_trap(rows)
    print(f"\n({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
