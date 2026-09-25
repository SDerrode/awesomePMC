"""
rerun_patch.py — the ``patch`` phase of ``rerun_pending.py`` (forecasting).

The rows of the rerun replace the corresponding rows of ``results/`` so that
``summarise.py`` rebuilds every table from one set of CSVs. The aggregates
are those of ``run_forecasting.py`` (``fc.aggregate`` on the same keys and
float formats); rows of other cases and methods are left as they are.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

import fc_common as fc
import run_forecasting as rf

HERE = Path(__file__).resolve().parent


def _replace(path: Path, new: pd.DataFrame, drop: pd.Series | np.ndarray, fmt: str) -> None:
    """Rewrite ``path`` with the rows ``drop`` removed and ``new`` appended (same float format)."""
    old = pd.read_csv(path)
    keep = old[~np.asarray(drop(old) if callable(drop) else drop, bool)]
    out = pd.concat([keep, new], ignore_index=True)
    out = out[list(old.columns) + [c for c in out.columns if c not in old.columns]]
    out.to_csv(path, index=False, float_format=fmt)


def _study_rows(rec: pd.DataFrame, rates: dict) -> pd.DataFrame:
    rec = rec.copy()
    rec["study"] = [rf.study_of(c) for c in rec.case]
    rec["variant"] = [c.split("__")[1] for c in rec.case]
    rec["rate"] = rec.case.map(rates)
    return fc.add_errors(rec)


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # pragma: no cover
        return "?"


def patch(args, sel, aoti, mote, sdf, cop, pmm) -> None:
    R = args.results
    t0 = time.perf_counter()
    cases = {cid: fc.build_case(cid, args.data, False) for cid in
             ["mote20__clean"] + [sp["case"] for sp, _, _ in aoti]}
    rates = {cid: c["rate"] for cid, c in cases.items()}

    # ---- mote-20 selection: the copula rows rerun, the HMC-IN rows kept
    old = pd.read_csv(R / "mote20_select.csv")
    sdf = sdf[list(old.columns) + [c for c in sdf.columns if c not in old.columns]]
    sdf.to_csv(R / "mote20_select.csv", index=False)

    # ---- records of the rerun
    recs = [res["records"] for _, res, _ in aoti + mote if "error" not in res]
    rec = _study_rows(pd.concat(recs, ignore_index=True), rates) if recs else None
    new_methods = {(c, m) for c, m in zip(rec.case, rec.method)} if rec is not None else set()

    def is_new(df, case_col="case"):
        return np.array([(c, m) in new_methods for c, m in zip(df[case_col], df.method)], bool)

    if rec is not None:
        # by (study, variant, method, h) and the part-3 aggregates, as run_forecasting
        key = {(rf.study_of(c), c.split("__")[1], m) for c, m in new_methods}
        _replace(R / "by_case_variant_h.csv", fc.aggregate(rec, ["study", "variant", "method", "h"]),
                 lambda o: [(s, v, m) in key for s, v, m in zip(o.study, o.variant, o.method)], "%.5g")
        _replace(R / "part3_real_by_h.csv", fc.aggregate(rec, ["study", "rate", "variant", "method", "h"]),
                 lambda o: [(s, v, m) in key for s, v, m in zip(o.study, o.variant, o.method)], "%.5g")
        _replace(R / "part3_real_by_case.csv", fc.aggregate(rec, ["study", "rate", "case", "method"]),
                 lambda o: is_new(o), "%.5g")
        # spike_last groups are pooled over the seeds of a rate: the rerun methods ran on seed 0 only
        sm = {(rf.study_of(c), rates[c], m) for c, m in new_methods}
        _replace(R / "part3_real_spike_last_h1.csv",
                 fc.aggregate(rec[rec.h == 1], ["study", "rate", "method", "spike_last"]),
                 lambda o: [(s, r, m) in sm for s, r, m in zip(o.study, o.rate, o.method)], "%.5g")

    # ---- part 1: the mote-20 copula PMC (clean case), paired against the PMM and the AR(1)
    paired_new = []
    if rec is not None and cop is not None and mote:
        m1 = rec[rec.case == "mote20__clean"]
        _replace(R / "part1_by_h.csv", fc.aggregate(m1, ["study", "method", "h"]),
                 lambda o: [(s, m) in {("mote20", x) for x in m1.method.unique()} for s, m in zip(o.study, o.method)],
                 "%.6g")
        _replace(R / "part1_all_h.csv", fc.aggregate(m1, ["study", "method"]),
                 lambda o: [(s, m) in {("mote20", x) for x in m1.method.unique()} for s, m in zip(o.study, o.method)],
                 "%.6g")
        _, bres, _ = rf_task_baselines(args)
        refs = [bres["records"]]
        if pmm is not None:
            refs.append(pmm)
        ref = _study_rows(pd.concat(refs, ignore_index=True), rates)
        # the recorded PMM, AR(1) and persistence scores are reproduced (all origins)
        p1 = pd.read_csv(R / "part1_by_h.csv")
        chk = {}
        for m in ("pmm|raw", "ar1|raw", "persistence|raw"):
            a = fc.aggregate(ref[ref.method == m], ["study", "method", "h"]).set_index("h").crps
            b = p1[(p1.study == "mote20") & (p1.method == m)].set_index("h").crps
            if len(a) and len(b):
                chk[m] = float(np.max(np.abs(a.loc[b.index] - b) / b))
        print("[patch] mote-20 references reproduced, max relative CRPS difference by h:", chk, flush=True)
        both = pd.concat([m1, ref], ignore_index=True)
        rng = np.random.default_rng(0)
        for m in sorted(m1.method.unique()):
            for b in ("pmm|raw", "ar1|raw", "persistence|raw"):
                if b in both.method.unique():
                    r = rf.paired(both, m, b, rng)
                    if r:
                        paired_new.append({"study": "mote20", **r})
        pn = pd.DataFrame(paired_new)
        _replace(R / "part1_paired.csv", pn,
                 lambda o: [(s, a, b) in {(x.study, x.a, x.b) for x in pn.itertuples()}
                            for s, a, b in zip(o.study, o.a, o.b)], "%.6g")
        args.reference_check = chk

    # ---- fits of the rerun Aotizhongxin tasks
    fits = [r for _, res, _ in aoti if "error" not in res for r in res["fits"]]
    if fits:
        fdf = pd.DataFrame(fits)
        _replace(R / "fits.csv", fdf,
                 lambda o: [(c, f) in {(x, y) for x, y in zip(fdf.case, fdf.family)} for c, f in zip(o.case, o.family)],
                 "%.6g")

    # ---- warnings and task times
    runs = sel + aoti + mote
    warn = pd.DataFrame([{"kind_task": sp.get("kind", ""), "K_task": sp.get("K", ""), **w}
                         for sp, res, _ in runs for w in res.get("warnings", [])])
    rerun_keys = {("pmc", sp["case"], sp["kind"]) for sp, _, _ in aoti} | {("select", "", k) for k, _ in
                                                                           [(sp["kind"], sp["K"]) for sp, _, _ in sel]}

    def old_task(o):
        return [(t, "" if c != c else c, k) in rerun_keys for t, c, k in zip(o.task, o.case, o.kind_task)]
    if len(warn):
        _replace(R / "pmcprg_warnings.csv", warn, old_task, "%.4g")
    tt = pd.DataFrame([{"task": sp["task"], "case": sp.get("case", ""), "kind": sp.get("kind", ""),
                        "K": sp.get("K", ""), "seconds": secs} for sp, _, secs in runs])
    _replace(R / "tasks.csv", tt,
             lambda o: [(("select" if t == "select" else t), "" if c != c else c, k) in rerun_keys
                        for t, c, k in zip(o.task, o.case, o.kind)], "%.3f")

    # ---- run_info.json: what was rerun, on which code
    info = json.loads((R / "run_info.json").read_text())
    qs = {}
    if rec is not None and "q_check" in rec.columns:
        g = rec[rec.q_check.notna()]
        qs = {"n_grid_rows": int(len(g)), "q_check_max": float(g.q_check.max()),
              "q_check_rows_above_0.25": int((g.q_check > 0.25).sum()),
              "q_check_tail_max": float(g.q_check_tail.max()),
              "q_check_tail_rows_above_0.05": int((g.q_check_tail > 0.05).sum())}
    ws = {}
    if len(warn):
        for kind, g in warn.groupby("kind"):
            ws[kind] = {"records": int(g["count"].sum()), "task_steps": int(len(g)),
                        "max_value": float(g.max_value.max()) if g.max_value.notna().any() else None}
    info["rerun_pending"] = {
        "date": time.strftime("%Y-%m-%d %H:%M"), "code": git_head(),
        "what": "rerun_pending.py: the mote-20 copula selection rows (ladder G = 64…512), the contaminated "
                "Aotizhongxin copula tasks (seed 0) and, if a copula model is chosen, its mote-20 forecasts",
        "mote20_copula_choice": list(cop) if cop else None,
        "mote20_origin_stride": getattr(args, "mote_stride", 1),
        "mote20_references_reproduced": getattr(args, "reference_check", None),
        "sum_task_s": float(sum(secs for _, _, secs in runs)),
        "errors": [f"{sp['task']} {sp.get('case', '')} {sp.get('kind', '')}: {res['error']}"
                   for sp, res, _ in runs if "error" in res],
        "pmcprg_warnings": ws, "quantile_check_stats": qs,
        "leading_gaps": {"mote20_selection_fits": [f"{r.kind} K={r.K} (τ max {r.max_diag_tau:.3f}, leading gap "
                                                   f"{int(r.lead_gap)})" for r in sdf.itertuples()
                                                   if r.kind != "hmc_in" and r.lead_gap > 0]},
    }
    (R / "run_info.json").write_text(json.dumps(info, indent=1, default=str))
    print(f"[patch] results/ updated in {time.perf_counter() - t0:.0f} s", flush=True)


def rf_task_baselines(args):
    """AR(1) and persistence on mote 20 (``fc.task_baselines``, a few seconds)."""
    return fc.run_task({"data_dir": args.data, "quick": False, "task": "baselines", "case": "mote20__clean"})
