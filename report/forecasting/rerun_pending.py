"""
rerun_pending.py — the cells of the forecasting study that waited on pmcprg's gap
quadrature, rerun on its final version (pmcprg d14e91d) (forecasting).

The campaign of ``run_forecasting.py`` ran on pmcprg 39f249f. Two groups of
cells waited on the gap quadrature (README, status):

* mote 20: the four copula candidates of the BIC choice (a leading gap of one
  row, Kendall τ up to 0.997); none converged in G at 256 nodes, so part 1 had
  no copula PMC for the mote;
* the five † fits of contaminated Aotizhongxin (copula PMC pair K = 3, seed 0),
  whose forecasts changed by 1.7–2.8 % of the predictive sd between G = 128
  and 256.

Phases (tasks cached in ``<out>/tasks`` under the key of ``run_forecasting``):

select  the four copula candidates, fitted as in phase 1 of run_forecasting
        (``fc.task_select``: the pipeline's record, ``fc.choose_nodes`` on
        G = 64 / 128 / 256), plus the ladder extended to 512: the checks of
        ``fc.quad_diff`` for 64 → 128, 128 → 256 and 256 → 512, quad_error /
        limit of the forecast chains at every G (``fc.quad_ratio``) and the
        log-likelihood of the training part at every G. Forecasts and
        posteriors are computed once per (origin, G) and task (``_Memo``).
aoti    the two contaminated Aotizhongxin copula tasks, ``fc.run_task`` with
        the specs of run_forecasting (raw, robust and Hampel fits, gated or
        not; ``fc.choose_nodes`` records the 128 → 256 check).
mote    the copula model the pipeline's rule chooses (if any; BIC among the
        fits that are neither degenerate nor unconverged), forecast on
        mote 20 at its G at every ``--mote-stride``-th origin (in chunks), with
        the AR(1), persistence and the PMM (``pmm_side.py refilter``: the
        recorded fits, no refit) on the same origins.
patch   the rerun rows replace the old ones in ``results/`` (fits.csv,
        mote20_select.csv, part1_*.csv, part3_real_*.csv, by_case_variant_h.csv,
        pmcprg_warnings.csv); run_info.json gets a ``rerun_pending`` entry.
        Then run summarise.py.

Usage (repository root)::

    PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python report/forecasting/rerun_pending.py \
        --phases select,aoti,mote,patch --jobs 6 --pmm-python <venv>/bin/python
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
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fc_common as fc  # noqa: E402
import run_forecasting as rf  # noqa: E402

REPO = HERE.parents[1]
#: The copula candidates of the mote-20 BIC choice (the HMC-IN rows have no grid).
COPULA = [(k, K) for k, K in fc.MOTE_CANDIDATES if k != "hmc_in"]
#: The node ladder of this rerun: run_forecasting's, then 512.
G_EXT = (64, 128, 256, 512)
AOTI_CASES = [f"aotizhongxin__spikes_r{r:.2f}_k{fc.SPIKE_K}_s0" for r in fc.REAL_RATES]


# ---------------------------------------------------------------------------
# Forecasts and posteriors computed once per (model, series, G) in a task
# ---------------------------------------------------------------------------

def _small_posterior(post) -> SimpleNamespace:
    """The fields of a GapPosterior this script reads (log_lik, quad_error)."""
    return SimpleNamespace(log_lik=float(post.log_lik),
                           quad_error=None if post.quad_error is None else float(post.quad_error))


class _Memo:
    """Memoises ``fc.forecast`` and ``fc.gap_posterior`` (and times them per G).

    Posteriors are kept reduced to (log_lik, quad_error) (``_small_posterior``).
    """

    def __init__(self):
        self.store: dict = {}
        self.secs: dict = {}
        self._orig = (fc.forecast, fc.gap_posterior)

    @staticmethod
    def _key(kind, model, Y, G, extra):
        mk = hashlib.sha1(json.dumps(model.raw, sort_keys=True, default=str).encode()).hexdigest()
        yk = hashlib.sha1(np.ascontiguousarray(Y, float).tobytes()).hexdigest()
        return kind, mk, yk, G, extra

    def _call(self, kind, fn, key, G):
        if key not in self.store:
            t = time.perf_counter()
            self.store[key] = fn()
            self.secs.setdefault(f"{kind}_{G}", []).append(time.perf_counter() - t)
        return self.store[key]

    def forecast(self, model, Y, h, *, gap_nodes=64, quantiles=(0.5,)):
        key = self._key("forecast", model, Y, gap_nodes, (h, tuple(quantiles)))
        return self._call("forecast", lambda: self._orig[0](model, Y, h, gap_nodes=gap_nodes, quantiles=quantiles),
                          key, gap_nodes)

    def gap_posterior(self, model, Y, *, gap_nodes=None, xi=True):
        # Only log_lik and quad_error are read here (quad_error_of, the ladder).
        # A full GapPosterior of mote 20 holds (M, K, G) arrays; keeping them for
        # every origin and G up to 512 took 17 GB in one worker.
        key = self._key("posterior", model, Y, gap_nodes, xi)
        return self._call("posterior", lambda: _small_posterior(
            self._orig[1](model, Y, gap_nodes=gap_nodes, xi=xi)), key, gap_nodes)

    def __enter__(self):
        fc.forecast, fc.gap_posterior = self.forecast, self.gap_posterior
        return self

    def __exit__(self, *a):
        fc.forecast, fc.gap_posterior = self._orig


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

def task_select_ext(spec: dict) -> dict:
    """``fc.task_select`` (the pipeline's record), then the ladder G = 64 … 512."""
    with _Memo() as memo:
        res = fc.task_select(spec)
        row = res["select"][0]
        model = fc.PMCModel.from_dict(next(iter(res["models"].values())))
        case = fc.build_case("mote20__clean", spec["data_dir"], spec["quick"])
        Y, origins, H = case["Y"], case["origins"], case["H"]
        Ytr = Y[:case["n_train"]]
        fam = fc.family(spec["kind"], spec["K"])
        for G, G2 in zip(G_EXT[:-1], G_EXT[1:]):
            fc.set_step(f"{fam}|select ladder {G}-{G2}")
            row[f"check_{G}_{G2}"], row[f"tail_{G}_{G2}"] = fc.quad_diff(model, Y, origins, H, G, G2)
        for G in G_EXT:
            fc.set_step(f"{fam}|select ladder ratio {G}")
            row[f"ratio_{G}"] = fc.quad_ratio(model, Y, origins, H, G)
            fc.set_step(f"{fam}|select ladder ll {G}")
            post = fc.gap_posterior(model, Ytr, gap_nodes=G, xi=False)
            row[f"ll_train_{G}"] = float(post.log_lik)
            row[f"quad_error_train_{G}"] = float(post.quad_error)
        row["seconds_by_G"] = json.dumps({k: round(float(np.mean(v)), 2) for k, v in memo.secs.items()})
    return res


def task_mote_fc(spec: dict) -> dict:
    """Forecasts of the chosen mote-20 copula model at a chunk of origins (``fc.pmc_rows``)."""
    fc._quiet()
    case = fc.build_case("mote20__clean", spec["data_dir"], spec["quick"])
    model = fc.PMCModel.from_dict(spec["model_raw"])
    all_o = list(case["origins"])
    o_mid = all_o[len(all_o) // 2]
    sub = dict(case, origins=np.asarray(spec["origins"]))
    fam = fc.family(spec["kind"], spec["K"])
    Yc = case["Y"]
    method = f"{fam}|raw"
    fits = []
    if spec.get("gated"):
        fc.set_step(f"{fam}|raw gate")
        Yc, f = fc.gate(model, Yc, spec["G"])
        method = f"{fam}|raw+gate"
        fits.append({"case": case["case"], "family": fam, "fit": "raw", "chunk": spec["chunk"],
                     **{f"gate_test_{k}": v for k, v in
                        fc.mask_stats(f, case, slice(case["n_train"], len(Yc))).items()}})
    fc.set_step(f"{fam}|{method.split('|')[1]} forecast")
    rows = fc.pmc_rows(method, model, Yc, sub, fc.spike_extra(case), spec["G"],
                       check_origin=o_mid if o_mid in spec["origins"] else None)
    df = pd.DataFrame(rows)
    df.insert(0, "case", case["case"])
    return {"records": df, "fits": fits}


TASKS = {"select_ext": task_select_ext, "mote_fc": task_mote_fc}


def run_task(spec: dict):
    if spec["task"] not in TASKS:
        return fc.run_task(spec)
    t0 = time.perf_counter()
    fc._quiet()
    fc.WARNINGS.take()
    try:
        res = TASKS[spec["task"]](spec)
    except Exception as exc:  # recorded; the other tasks go on
        import traceback
        res = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-2000:]}
    res["warnings"] = [{"task": spec["task"], "case": spec.get("case", ""), **w} for w in fc.WARNINGS.take()]
    return spec, res, time.perf_counter() - t0


def run_all(specs: list[dict], out: Path, jobs: int, label: str) -> list[tuple]:
    """``run_forecasting.run_all`` with this module's tasks (same cache keys)."""
    cache = out / "tasks"
    cache.mkdir(parents=True, exist_ok=True)
    todo, done = [], []
    for sp in specs:
        f = cache / f"{rf.spec_key(sp)}.pkl"
        if f.exists():
            with f.open("rb") as fh:
                done.append(pickle.load(fh))
        else:
            todo.append(sp)
    print(f"[{label}] {len(specs)} task(s), {len(done)} cached, {len(todo)} to run on {jobs} process(es)",
          flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max(1, min(jobs, len(todo)))) as ex:
        futs = [ex.submit(run_task, sp) for sp in todo]
        for fut in as_completed(futs):
            sp, res, secs = item = fut.result()
            if "error" in res:
                print(f"   ERROR {sp['task']} {sp.get('case', '')} {sp.get('kind', '')}: {res['error']}\n"
                      f"{res['traceback']}", flush=True)
            else:
                with (cache / f"{rf.spec_key(sp)}.pkl").open("wb") as fh:
                    pickle.dump(item, fh)
            done.append(item)
            print(f"   {len(done)}/{len(specs)} {sp['task']} {sp.get('case', '')} {sp.get('kind', '')}"
                  f"{sp.get('K', '')} {sp.get('chunk', '')} {secs:.0f} s  [{time.perf_counter() - t0:.0f} s]",
                  flush=True)
    return done


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def select_specs(base: dict) -> list[dict]:
    return [{**base, "task": "select_ext", "kind": k, "K": K} for k, K in COPULA]


def aoti_specs(base: dict) -> list[dict]:
    # the specs of run_forecasting phase 2 (same cache keys)
    return [{**base, "task": "pmc", "case": cid, "kind": "pmc_pair", "K": 3, "hampel": True} for cid in AOTI_CASES]


def choose(sel: list[tuple], results: Path) -> tuple[pd.DataFrame, dict, tuple | None]:
    """The pipeline's rule on the rerun copula rows and the recorded HMC-IN rows."""
    rows = [r for _, res, _ in sel if "error" not in res for r in res["select"]]
    models = {k: v for _, res, _ in sel if "error" not in res for k, v in res["models"].items()}
    old = pd.read_csv(results / "mote20_select.csv")
    sdf = pd.concat([pd.DataFrame(rows), old[old.kind == "hmc_in"]], ignore_index=True).sort_values("bic")
    ok = sdf[~sdf.degenerate.astype(bool) & (sdf.quad_check <= fc.QUAD_TOL)
             & (sdf.quad_check_tail <= fc.QUAD_TOL_TAIL)]
    chosen = []
    for pool in (ok[ok.kind == "hmc_in"], ok[ok.kind != "hmc_in"]):
        if len(pool):
            r = pool.iloc[0]
            chosen.append((r.kind, int(r.K)))
    sdf["chosen"] = [(k, int(K)) in chosen for k, K in zip(sdf.kind, sdf.K)]
    cop = [c for c in chosen if c[0] != "hmc_in"]
    return sdf, models, (cop[0] if cop else None)


def mote_specs(base: dict, sdf: pd.DataFrame, models: dict, cop: tuple, stride: int, chunks: int,
               gated: bool) -> list[dict]:
    case = fc.build_case("mote20__clean", base["data_dir"], base["quick"])
    origins = [int(o) for o in case["origins"]][::stride]
    kind, K = cop
    G = int(sdf[(sdf.kind == kind) & (sdf.K == K)].gap_nodes.iloc[0])
    raw = models[f"mote20__{fc.family(kind, K)}"]
    specs = []
    for g in ([False, True] if gated else [False]):
        for c in range(chunks):
            specs.append({**base, "task": "mote_fc", "kind": kind, "K": K, "G": G, "gated": g, "chunk": c,
                          "origins": origins[c::chunks], "model_raw": raw,
                          "model_sha": hashlib.sha1(json.dumps(raw, sort_keys=True).encode()).hexdigest()[:12]})
    return specs


def pmm_mote(args, case: dict) -> pd.DataFrame | None:
    """PMM forecasts on mote 20 from the recorded fits (``pmm_side.py refilter``)."""
    jobdir = args.out / "pmm_jobs"
    rf.write_pmm_jobs({"mote20__clean": case}, jobdir)
    f_csv = jobdir / "mote20__clean.pmm.csv"
    if not f_csv.exists():
        if not args.pmm_python:
            print("[pmm] no --pmm-python: PMM rows skipped", flush=True)
            return None
        cmd = [args.pmm_python, "-B", str(HERE / "pmm_side.py"), "refilter", "--dir", str(jobdir),
               "--fits", str(args.results / "pmm_fits.csv")]
        print("[pmm]", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=REPO, check=True)
    return fc.pmm_rows(case, pd.read_csv(f_csv))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=fc.DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=REPO / "report" / "out" / "forecasting" / "pending")
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--phases", default="select,aoti,mote,patch")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--mote-stride", type=int, default=1, help="forecast every k-th mote-20 origin")
    ap.add_argument("--mote-chunks", type=int, default=4)
    ap.add_argument("--mote-gated", action="store_true", help="also the gated forecasts of the mote copula PMC")
    ap.add_argument("--pmm-python", default=os.environ.get("PMMFORECAST_PYTHON"))
    args = ap.parse_args(argv)
    phases = set(args.phases.split(","))
    args.out.mkdir(parents=True, exist_ok=True)
    base = {"data_dir": args.data, "quick": False}
    T0 = time.perf_counter()
    specs = []
    if "select" in phases or "mote" in phases or "patch" in phases:
        specs += select_specs(base)
    if "aoti" in phases or "patch" in phases:
        specs += aoti_specs(base)
    # the select and aoti tasks share the pool (the aoti tasks are the longest: first)
    specs.sort(key=lambda s: s["task"] != "pmc")
    done = run_all(specs, args.out, args.jobs, "select + aoti")
    sel = [d for d in done if d[0]["task"] == "select_ext"]
    aoti = [d for d in done if d[0]["task"] == "pmc"]
    sdf, models, cop = choose(sel, args.results)
    sdf.to_csv(args.out / "mote20_select.csv", index=False)
    print("[select mote20] copula choice:", cop, flush=True)
    cols = ["kind", "K", "ll", "bic", "degenerate", "gap_nodes", "quad_check", "quad_check_tail",
            "check_64_128", "check_128_256", "check_256_512", "tail_256_512",
            "ratio_64", "ratio_128", "ratio_256", "ratio_512", "chosen"]
    print(sdf[[c for c in cols if c in sdf.columns]].to_string(), flush=True)
    mote = []
    if "mote" in phases and cop is not None:
        mote = run_all(mote_specs(base, sdf, models, cop, args.mote_stride, args.mote_chunks, args.mote_gated),
                       args.out, args.jobs, "mote20 copula forecasts")
    with (args.out / "rerun.pkl").open("wb") as fh:
        pickle.dump({"sel": sel, "aoti": aoti, "mote": mote, "sdf": sdf, "cop": cop,
                     "stride": args.mote_stride}, fh)
    if "patch" in phases:
        import rerun_patch
        case = fc.build_case("mote20__clean", args.data, False)
        pmm = pmm_mote(args, case) if cop is not None and mote else None
        rerun_patch.patch(args, sel, aoti, mote, sdf, cop, pmm)
    print(f"done in {time.perf_counter() - T0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
