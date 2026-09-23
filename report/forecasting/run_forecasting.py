"""
run_forecasting.py — forecasting study: pmcprg vs pmmforecast, and what outlier
gating does to forecasts (forecasting).

Part 1 (clean cases) — rolling-origin h-step forecasts on three real series:
Beijing Aotizhongxin (as report/real_series §4: dequantised ln PM2.5, first 75 %
for fitting, 91 daily origins, h = 1…24, scored against ln PM2.5 as reported),
tsNH4 (complete series, ln NH4, 10 min) and Intel Lab mote 20 (temperature,
30 s, epochs 1…28 800, suspect readings removed). Models: pmcprg PMC and HMC-IN
(real_series choices; BIC on the training part for the mote), pmmforecast's
Gaussian PMM (Y-only Kalman MLE, ``pmm_side.py``), AR(1), persistence.

Part 3 (contaminated cases) — the same pipelines with raw fits, sequential
gating of the conditioning data (``flag_outliers``, flagged rows → NaN) and
``robust_estimate``, on the report/erroneous_data fixtures (N = 3000, spikes
of 6 sd at 0 / 1 / 5 %, 10 replicates), on Aotizhongxin with injected spikes
(6 sd, 1 / 5 %, 3 seeds) and on mote 20 as recorded (3 suspect readings of
−38.4 °C in the training part).

STATUS (2026-09-22): the full campaign was stopped on three pmcprg problems
that make parts of its output silently wrong (README, "Library problems"
P1–P3). The driver detects P1 (per-model quadrature check, ``choose_nodes``)
and P3 (``q_check`` of every grid forecast, ``quantile_inconsistent_rows`` in
run_info.json); rerun it once they are fixed. Part 2 (``cross_check.py``) is
not affected.

Phases: (0) pmmforecast jobs are written and ``pmm_side.py jobs`` is started in
the background with ``--pmm-python`` (skipped with a message otherwise);
(1) mote-20 BIC choice; (2) pmcprg tasks (cached in ``<out>/tasks``);
(3) PMM forecasts scored, everything aggregated into ``results/``.

Usage (repository root)::

    PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python report/forecasting/run_forecasting.py \
        --quick --pmm-python <venv>/bin/python          # smoke run → report/out/forecasting/quick
    PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python report/forecasting/run_forecasting.py \
        --jobs 5 --pmm-python <venv>/bin/python --pmm-workers 4
    .venv/bin/python report/forecasting/summarise.py
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fc_common as fc  # noqa: E402

REPO = HERE.parents[1]
_VOLATILE = ("data_dir", "quick", "out")
N_BOOT = 2000


# ---------------------------------------------------------------------------
# Task cache
# ---------------------------------------------------------------------------

def spec_key(spec: dict) -> str:
    s = json.dumps({k: v for k, v in spec.items() if k not in _VOLATILE and k != "model_raw"},
                   sort_keys=True, default=str)
    return f"{spec['task']}_{hashlib.sha1(s.encode()).hexdigest()[:12]}"


def run_all(specs: list[dict], out: Path, jobs: int, force: bool, label: str) -> list[tuple]:
    cache = out / "tasks"
    cache.mkdir(parents=True, exist_ok=True)
    todo, done = [], []
    for sp in specs:
        f = cache / f"{spec_key(sp)}.pkl"
        if f.exists() and not force:
            with f.open("rb") as fh:
                done.append(pickle.load(fh))
        else:
            todo.append(sp)
    print(f"[{label}] {len(specs)} task(s), {len(done)} cached, {len(todo)} to run on {jobs} process(es)",
          flush=True)
    t0 = time.perf_counter()

    def store(item):
        sp, res, secs = item
        if "error" in res:
            print(f"   ERROR {sp['task']} {sp.get('case', '')} {sp.get('kind', '')}: {res['error']}\n"
                  f"{res['traceback']}", flush=True)
        else:
            with (cache / f"{spec_key(sp)}.pkl").open("wb") as fh:
                pickle.dump(item, fh)
        done.append(item)
        print(f"   {len(done)}/{len(specs)} {sp['task']} {sp.get('case', '')} {sp.get('kind', '')}"
              f"{sp.get('K', '')} {secs:.0f} s  [{time.perf_counter() - t0:.0f} s]", flush=True)

    if jobs <= 1:
        for sp in todo:
            store(fc.run_task(sp))
    else:
        with ProcessPoolExecutor(jobs) as ex:
            futs = [ex.submit(fc.run_task, sp) for sp in todo]
            for f in as_completed(futs):
                store(f.result())
    return done


# ---------------------------------------------------------------------------
# pmmforecast side (separate interpreter)
# ---------------------------------------------------------------------------

def write_pmm_jobs(cases: dict, jobdir: Path) -> None:
    jobdir.mkdir(parents=True, exist_ok=True)
    for cid, case in cases.items():
        spec = {"name": cid, "format": 2, "n_train": int(case["n_train"]),
                "origins": [int(o) for o in case["origins"]],
                "H": int(case["H"]), "n_restarts": fc.PMM_RESTARTS, "seed": fc.PMM_SEED,
                "theory_n0": fc.PMM_THEORY_N0}
        js = jobdir / f"{cid}.json"
        new = json.dumps(spec, indent=1)
        csv = jobdir / f"{cid}.csv"
        ycsv = pd.DataFrame({"y": case["Y"]})
        if js.exists() and js.read_text() == new and csv.exists():
            old = pd.read_csv(csv)["y"].to_numpy()
            if old.shape == case["Y"].shape and np.array_equal(np.isnan(old), np.isnan(case["Y"])) \
                    and np.allclose(np.nan_to_num(old), np.nan_to_num(case["Y"]), rtol=0, atol=1e-12):
                continue
            for suffix in (".pmm.json", ".pmm.csv"):   # stale output
                (jobdir / f"{cid}{suffix}").unlink(missing_ok=True)
        js.write_text(new)
        ycsv.to_csv(csv, index=False, float_format="%.17g")


def start_pmm(args, jobdir: Path):
    if not args.pmm_python:
        print("[pmm] no --pmm-python (or PMMFORECAST_PYTHON): pmmforecast is not run; the PMM "
              "columns are filled from existing outputs in "
              f"{jobdir} if any, skipped otherwise.", flush=True)
        return None
    log = (args.out / "pmm_side.log").open("w")
    cmd = [args.pmm_python, "-B", str(HERE / "pmm_side.py"), "jobs", "--dir", str(jobdir),
           "--workers", str(args.pmm_workers)]
    print("[pmm] started:", " ".join(cmd), flush=True)
    return subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT), log


def wait_pmm(proc) -> bool:
    if proc is None:
        return False
    p, log = proc
    t0 = time.perf_counter()
    rc_ = p.wait()
    log.close()
    print(f"[pmm] pmm_side.py finished with status {rc_} (waited {time.perf_counter() - t0:.0f} s)", flush=True)
    if rc_ == 3:
        print("[pmm] pmmforecast is not importable in that interpreter: PMM part skipped.", flush=True)
    return rc_ == 0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def study_of(case: str) -> str:
    s = case.split("__")[0]
    return s[4:] if s.startswith("fix_") else s


def paired(df: pd.DataFrame, a: str, b: str, rng: np.random.Generator) -> dict | None:
    """Mean over origins of (CRPS_a − CRPS_b), per-origin means over h; bootstrap 95 % CI."""
    pa = df[df.method == a].groupby("origin").crps.mean()
    pb = df[df.method == b].groupby("origin").crps.mean()
    j = pa.index.intersection(pb.index)
    if len(j) < 5:
        return None
    d = (pa.loc[j] - pb.loc[j]).to_numpy()
    boots = rng.choice(d, size=(N_BOOT, d.size), replace=True).mean(axis=1)
    return {"a": a, "b": b, "n_origins": int(d.size), "mean_diff": float(d.mean()),
            "ci_lo": float(np.quantile(boots, 0.025)), "ci_hi": float(np.quantile(boots, 0.975)),
            "frac_a_better": float(np.mean(d < 0))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=fc.DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--results", type=Path, default=None)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--pmm-python", default=os.environ.get("PMMFORECAST_PYTHON"))
    ap.add_argument("--pmm-workers", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    tag = "quick" if args.quick else "full"
    args.out = args.out or REPO / "report" / "out" / "forecasting" / tag
    args.results = args.results or (args.out / "results" if args.quick else HERE / "results")
    args.out.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)
    if args.quick:
        fc.FIX_REPS, fc.REAL_SEEDS = 1, 1
    T0 = time.perf_counter()

    # ---- cases and pmmforecast jobs
    ids = fc.case_ids()
    cases = {cid: fc.build_case(cid, args.data, args.quick) for cid in ids}
    if args.quick:   # the NaN-aware Hampel equals report/erroneous_data's on a NaN-free series
        sys.path.insert(0, str(REPO / "report" / "erroneous_data"))
        from run_study import hampel as hampel_ref
        y = cases[f"fix_pmc_gauss_k2__r0.05_k{fc.SPIKE_K}_rep0"]["Y"]
        assert np.array_equal(hampel_ref(y), fc.hampel_nan(y)), "hampel_nan differs from the reference"
        print("[check] hampel_nan == report/erroneous_data hampel on a NaN-free series", flush=True)
        # The closed-form AR(1) gate equals flag_outliers on the AR(1) as a pmcprg model
        # where the quadrature is accurate (φ ≈ 0.95 here; with gaps and spikes).
        from pmcprg.pmc import flag_outliers
        c = cases[f"aotizhongxin__spikes_r0.05_k{fc.SPIKE_K}_s0"]
        par = fc.rc.ar1_fit(c["Y"][:c["n_train"]])
        ref = flag_outliers(fc.ar1_model(par), c["Y"], alpha=fc.ALPHA).flagged
        mine = fc.ar1_flags(c["Y"], par)
        assert np.array_equal(ref, mine), f"ar1_flags differs from flag_outliers ({(ref != mine).sum()} rows)"
        print(f"[check] ar1_flags == flag_outliers(ar1_model) (φ = {par['phi']:.3f}, {mine.sum()} flags)", flush=True)
    jobdir = args.out / "pmm_jobs"
    write_pmm_jobs(cases, jobdir)
    proc = start_pmm(args, jobdir)
    base = {"data_dir": args.data, "quick": args.quick}

    # ---- phase 1: mote-20 model choice (BIC on the clean training part)
    sel = run_all([{**base, "task": "select", "kind": k, "K": K} for k, K in fc.MOTE_CANDIDATES],
                  args.out, args.jobs, args.force, "select mote20")
    select_rows = [r for _, res, _ in sel if "error" not in res for r in res["select"]]
    mote_models = {k: v for _, res, _ in sel if "error" not in res for k, v in res["models"].items()}
    sdf = pd.DataFrame(select_rows).sort_values("bic")
    # Degenerate fits (real_series rule) and fits whose forecasts are not converged in
    # the quadrature (the pmcprg problem of README "Library problems") are not chosen.
    ok = sdf[~sdf.degenerate.astype(bool) & (sdf.quad_check <= fc.QUAD_TOL)]
    chosen = []
    for pool in (ok[ok.kind == "hmc_in"], ok[ok.kind != "hmc_in"]):
        if len(pool):
            r = pool.iloc[0]
            chosen.append((r.kind, int(r.K)))
    sdf["chosen"] = [(k, int(K)) in chosen for k, K in zip(sdf.kind, sdf.K)]
    sdf.to_csv(args.results / "mote20_select.csv", index=False)
    print("[select mote20] chosen:", chosen, flush=True)
    fc.REAL["mote20"]["models"] = chosen

    # ---- phase 2: pmcprg tasks
    specs = []
    for cid, case in cases.items():
        if cid.startswith("fix_"):
            specs.append({**base, "task": "fixture", "case": cid})
            continue
        series, variant = cid.split("__")
        for kind, K in fc.REAL[series]["models"]:
            sp = {**base, "task": "pmc", "case": cid, "kind": kind, "K": K, "hampel": variant != "clean"}
            if cid == "mote20__clean":
                sp["model_raw"] = mote_models[f"mote20__{fc.family(kind, K)}"]
            specs.append(sp)
        specs.append({**base, "task": "baselines", "case": cid})
    # longest first
    order = {"pmc": 0, "fixture": 1, "baselines": 2}
    specs.sort(key=lambda s: (order[s["task"]], not s.get("case", "").startswith("mote20"),
                              s.get("kind", "") != "pmc_pair"))
    done = run_all(specs, args.out, args.jobs, args.force, "pmcprg")
    errors = [(sp, res) for sp, res, _ in done if "error" in res]
    recs = [res["records"] for _, res, _ in done if "error" not in res]
    fits = [r for _, res, _ in done if "error" not in res for r in res["fits"]]
    task_times = [{"task": sp["task"], "case": sp.get("case", ""), "kind": sp.get("kind", ""),
                   "K": sp.get("K", ""), "seconds": secs} for sp, _, secs in done + sel]

    # ---- phase 3: pmmforecast forecasts
    have_pmm = wait_pmm(proc) if proc is not None else False
    pmm_info = []
    for cid, case in cases.items():
        f_csv, f_json = jobdir / f"{cid}.pmm.csv", jobdir / f"{cid}.pmm.json"
        if f_csv.exists() and f_json.exists():
            have_pmm = True
            recs.append(fc.pmm_rows(case, pd.read_csv(f_csv)))
            pmm_info.append(json.loads(f_json.read_text()))
    if not have_pmm:
        print("[pmm] no pmmforecast output: PMM columns skipped.", flush=True)
    rec = pd.concat(recs, ignore_index=True)
    rec["study"] = [study_of(c) for c in rec.case]
    rec["variant"] = [c.split("__")[1] for c in rec.case]
    rec = fc.add_errors(rec)
    rec.to_pickle(args.out / "records.pkl.gz")

    # ---- aggregation
    R = args.results
    real_clean = rec[rec.case.isin([f"{s}__clean" for s in fc.REAL])]
    p1 = fc.aggregate(real_clean, ["study", "method", "h"])
    p1.to_csv(R / "part1_by_h.csv", index=False, float_format="%.6g")
    p1all = fc.aggregate(real_clean, ["study", "method"])
    p1all.to_csv(R / "part1_all_h.csv", index=False, float_format="%.6g")
    # paired CRPS differences (per-origin means over h) against pmmforecast and between pmcprg models
    rng = np.random.default_rng(0)
    pr = []
    for study, g in real_clean.groupby("study"):
        meths = sorted(g.method.unique())
        refs = [m for m in ("pmm|raw", "ar1|raw") if m in meths]
        for ref in refs:
            for m in meths:
                if m != ref and not (ref == "ar1|raw" and m == "pmm|raw"):
                    r = paired(g, m, ref, rng)
                    if r:
                        pr.append({"study": study, **r})
    pd.DataFrame(pr).to_csv(R / "part1_paired.csv", index=False, float_format="%.6g")
    # theoretical vs empirical MSE of the PMM
    th = []
    for info in pmm_info:
        cid = info["name"]
        if cid not in [f"{s}__clean" for s in fc.REAL]:
            continue
        study = cid.split("__")[0]
        e = p1[(p1.study == study) & (p1.method == "pmm|raw")].set_index("h")
        for k, t in enumerate(info["theory_y_mse"], start=1):
            th.append({"study": study, "h": k, "theory_mse": t,
                       "steady_pred_var": info["steady_pred_var"][k - 1] if info["steady_pred_var"] else np.nan,
                       "emp_mse": float(e.loc[k, "mse"]) if k in e.index else np.nan,
                       "emp_mean_pred_var": float((real_clean[(real_clean.study == study) & (real_clean.method == "pmm|raw")
                                                              & (real_clean.h == k)].sd ** 2).mean())})
    pd.DataFrame(th).to_csv(R / "part1_pmm_theory.csv", index=False, float_format="%.6g")
    # part 3: contaminated cases (and the clean references), averaged over replicates
    p3 = fc.aggregate(rec, ["study", "variant", "method", "h"])
    p3.to_csv(R / "by_case_variant_h.csv", index=False, float_format="%.5g")
    rate_of = {cid: case["rate"] for cid, case in cases.items()}
    rec["rate"] = rec.case.map(rate_of)
    fx = rec[rec.case.str.startswith("fix_")]
    fx3 = fc.aggregate(fx, ["study", "rate", "method", "h"])
    fx3.to_csv(R / "part3_fixtures_by_h.csv", index=False, float_format="%.5g")
    fxs = fc.aggregate(fx[fx.h == 1], ["study", "rate", "method", "spike_last"])
    fxs.to_csv(R / "part3_fixtures_spike_last_h1.csv", index=False, float_format="%.5g")
    fxc = fc.aggregate(fx, ["study", "rate", "case", "method"])
    fxc.to_csv(R / "part3_fixtures_by_case.csv", index=False, float_format="%.5g")
    rl = rec[~rec.case.str.startswith("fix_")]
    rl3 = fc.aggregate(rl, ["study", "rate", "variant", "method", "h"])
    rl3.to_csv(R / "part3_real_by_h.csv", index=False, float_format="%.5g")
    rls = fc.aggregate(rl[rl.h == 1], ["study", "rate", "method", "spike_last"])
    rls.to_csv(R / "part3_real_spike_last_h1.csv", index=False, float_format="%.5g")
    rlc = fc.aggregate(rl, ["study", "rate", "case", "method"])
    rlc.to_csv(R / "part3_real_by_case.csv", index=False, float_format="%.5g")
    pd.DataFrame(fits).to_csv(R / "fits.csv", index=False, float_format="%.6g")
    if pmm_info:
        pdf = pd.DataFrame(pmm_info)
        for col in ("theory_y_mse", "steady_pred_var", "abcde"):
            pdf[col] = [" ".join(f"{v:.6g}" for v in x) if isinstance(x, list) else "" for x in pdf[col]]
        pdf.to_csv(R / "pmm_fits.csv", index=False, float_format="%.6g")
    pd.DataFrame(task_times).to_csv(R / "tasks.csv", index=False, float_format="%.3f")
    # quantile CRPS (200 levels) against the exact CRPS where it exists
    # (Gaussian laws: AR(1), PMM; Gaussian mixtures: HMC-IN)
    crps_chk = {}
    for lab, sel in (("gaussian", rec.method.str.startswith(("ar1|", "pmm|", "pmm_default|"))),
                     ("hmc_in_mixture", rec.method.str.startswith("hmc_in_"))):
        d = rec.loc[sel]
        if len(d):
            rel = (d.crps_q - d.crps).abs() / d.crps.clip(lower=1e-12)
            crps_chk[lab] = {"n": int(len(d)), "mean_rel_diff": float(rel.mean()),
                             "max_rel_diff": float(rel.max()),
                             "rel_diff_of_mean": float(abs(d.crps_q.mean() - d.crps.mean()) / d.crps.mean())}
    # Rows whose returned quantiles disagree with the forecast's own node law
    # (pmcprg problem P3 of the README): CRPS and coverage of those rows are wrong.
    q_bad = {}
    if "q_check" in rec.columns:
        q_bad = {m: int(n) for m, n in rec[rec.q_check > 0.25].groupby("method").size().items()}
        if q_bad:
            print(f"WARNING: {sum(q_bad.values())} score rows have quantiles inconsistent with the "
                  f"forecast's node law (pmcprg problem P3): {q_bad}", flush=True)
    info = {
        "quantile_inconsistent_rows": q_bad,
        "crps_quantile_vs_exact": crps_chk,
        "date": time.strftime("%Y-%m-%d %H:%M"), "quick": args.quick, "jobs": args.jobs,
        "pmm_workers": args.pmm_workers, "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__, "machine": platform.machine(),
        "system": platform.system(), "data": args.data, "n_records": int(len(rec)),
        "n_errors": len(errors), "errors": [f"{sp['task']} {sp.get('case', '')}: {res['error']}" for sp, res in errors],
        "wall_s": time.perf_counter() - T0,
        "sum_task_s": float(sum(t["seconds"] for t in task_times)),
        "pmm_sum_s": float(sum(i["seconds"] for i in pmm_info)) if pmm_info else None,
        "settings": {k: getattr(fc, k) for k in ("ALPHA", "HAMPEL_HALF", "HAMPEL_T", "SPIKE_K", "PMM_RESTARTS",
                                                  "PMM_SEED", "PMM_THEORY_N0", "FIX_N_TRAIN", "FIX_N_TEST",
                                                  "FIX_H", "FIX_STEP", "FIX_RATES", "FIX_REPS", "REAL_RATES",
                                                  "REAL_SEEDS", "MOTE_ID", "MOTE_EPOCHS")},
        "real": {k: {kk: vv for kk, vv in v.items()} for k, v in fc.REAL.items()},
        "n_quantile_levels_crps": int(len(fc.TAUS)),
    }
    (R / "run_info.json").write_text(json.dumps(info, indent=1, default=str))
    print(f"done in {time.perf_counter() - T0:.0f} s; {len(rec)} score rows; {len(errors)} task error(s)", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
