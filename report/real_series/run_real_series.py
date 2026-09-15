"""
run_real_series.py — A4: unsupervised PMC/HMC study on three real series with gaps.

Series (read-only, outside the repository; ``--data``):

* tsNH4 (imputeTS, log scale): real gaps with ground truth.
* Beijing PM2.5, Huairou and Aotizhongxin one-year extracts (dequantised
  log): real gaps without truth; masked cross-validation.
* PAMAP2 subjects 102, 108, 105: 2 Hz log-SD of the hand accelerometer norm,
  activity labels grouped into K = 2 and K = 3 intensity classes; real
  (state-dependent) sensor dropouts plus added MCAR blocks.

Phases
------
1. Data summaries, baselines at the real gaps, supervised oracles and every
   unsupervised fit (ICE "available", ICE "impute", SEM; k-means start and
   library multistart draws), each with its evaluations.
2. Masked cross-validation (Beijing) and rolling forecasts, on the models
   chosen by BIC in phase 1.
3. Aggregation: result CSVs, stability columns, small figure extracts.

Every task is keyed by its parameters and cached in ``<out>/tasks`` (rerun
skips finished tasks; ``--force`` recomputes). Seeds are fixed in the task
specs: the results do not depend on ``--jobs``.

Usage (repository root)::

    PYTHONPATH=. .venv/bin/python report/real_series/run_real_series.py --quick
    PYTHONPATH=. .venv/bin/python report/real_series/run_real_series.py --jobs 3
    PYTHONPATH=. .venv/bin/python report/real_series/summarise.py
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
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rs_common as rc  # noqa: E402

REPO = HERE.parent.parent
SUBJECTS = (102, 108, 105)
PAMAP2_MODELS = ("hmc_in", "pmc_state", "pmc_pair")
ALL_MODELS = ("hmc_in", "pmc_state", "pmc_pair", "pmc_state_gice")
CV_RATES = (0.05, 0.20)
CV_BLOCKS = (1, 6, 48)
CV_SEEDS = 5


# ---------------------------------------------------------------------------
# Task lists
# ---------------------------------------------------------------------------

def _plan_full(series: str, K: int, model: str) -> dict[str, int]:
    """Strategies and number of starts of a fit cell (full campaign)."""
    if series.endswith("_complete"):
        return {"complete": 3}
    if series in ("pamap2_108_gapped", "pamap2_105_gapped"):
        return {"available": 3}
    if series == "aotizhongxin":
        return {"available": 1}
    if series == "huairou_rawlog":
        return {"available": 1}
    if model == "pmc_state_gice":
        return {"available": 3, "impute": 1, "sem": 1}
    if series == "tsnh4" or K == 2 or model == "hmc_in":
        return {"available": 3, "impute": 3, "sem": 3}
    return {"available": 3, "impute": 1, "sem": 1}


def fit_cells(quick: bool) -> list[tuple[str, int, str, dict, tuple]]:
    """(series, K, model, {strategy: n_starts}, evals)."""
    cells = []
    if quick:
        for m in ("hmc_in", "pmc_state"):
            cells.append(("tsnh4", 2, m, {"available": 2, "impute": 1, "sem": 1}, ("tsnh4_impute",)))
        for m in ("hmc_in", "pmc_state"):
            cells.append(("aotizhongxin", 2, m, {"available": 1}, ()))
        cells.append(("pamap2_105_complete", 2, "hmc_in", {"complete": 1}, ("pamap2",)))
        cells.append(("pamap2_105_gapped", 2, "hmc_in", {"available": 1}, ("pamap2",)))
        return cells
    for K in (2, 3):
        for m in ALL_MODELS:
            cells.append(("tsnh4", K, m, _plan_full("tsnh4", K, m), ("tsnh4_impute",)))
            cells.append(("huairou", K, m, _plan_full("huairou", K, m), ()))
        for m in PAMAP2_MODELS:
            cells.append(("aotizhongxin", K, m, _plan_full("aotizhongxin", K, m), ()))
        for m in ("hmc_in", "pmc_state"):
            cells.append(("huairou_rawlog", K, m, _plan_full("huairou_rawlog", K, m), ()))
        for subj in SUBJECTS:
            for m in PAMAP2_MODELS:
                for which in ("complete", "gapped"):
                    s = f"pamap2_{subj}_{which}"
                    cells.append((s, K, m, _plan_full(s, K, m), ("pamap2",)))
    return cells


#: Rough relative cost, used only to start the long tasks first.
_COST = {"hmc_in": 1, "pmc_state": 15, "pmc_pair": 25, "pmc_state_gice": 40}
_STRAT_COST = {"available": 1, "complete": 1, "sem": 1.5, "impute": 5}


def phase1_specs(quick: bool) -> list[dict]:
    subjects = (105,) if quick else SUBJECTS
    specs: list[dict] = []
    for subj in subjects:
        specs.append({"task": "pamap2_missingness", "subject": subj, "save_strip": subj == subjects[0]})
        specs.append({"task": "mnar", "subject": subj})
        specs.append({"task": "pamap2_feature_check", "subject": subj})
        for K in ((2,) if quick else (2, 3)):
            for m in (("hmc_in",) if quick else PAMAP2_MODELS):
                specs.append({"task": "oracle", "subject": subj, "K": K, "model": m})
    specs.append({"task": "tsnh4_baselines"})
    for series, K, model, plan, evals in fit_cells(quick):
        n_starts = max(plan.values())
        for strategy, n in plan.items():
            for start in range(n):
                specs.append({"task": "fit", "series": series, "K": K, "model": model,
                              "strategy": strategy, "start": start, "n_starts": n_starts,
                              "evals": list(evals)})
    return specs


def _cost(spec: dict) -> float:
    if spec["task"] == "fit":
        c = _COST[spec["model"]] * _STRAT_COST[spec["strategy"]] * (1.5 if spec["K"] == 3 else 1)
        return c * (0.5 if spec["series"] == "tsnh4" else 1)
    if spec["task"] == "cv":
        return 10 if spec.get("model", "hmc_in") != "hmc_in" else 1
    return 2


def is_degenerate(r, args) -> bool:
    """A margin sd below DEGENERATE_SD_RATIO × sd of the series, or a state with π < DEGENERATE_MIN_PI."""
    raw = json.loads((args.out / "models" / f"{rc.fit_tag(r)}.json").read_text())
    y_sd = float(np.nanstd(rc.series_data(r.series, args.data, args.quick)))
    mdl = rc.PMCModel.from_dict(raw)
    return bool(rc.min_margin_sd(mdl) < rc.DEGENERATE_SD_RATIO * y_sd
                or np.min(mdl.stationary_pi) < rc.DEGENERATE_MIN_PI)


def choose_models(fits: pd.DataFrame, series: str, args) -> list[dict]:
    """HMC-IN and the best Gaussian-margin PMC, each at its BIC-best K (ICE "available").

    Degenerate fits (:func:`is_degenerate`) are excluded; GICE fits are left out
    of the refits of phase 2 (cost).
    """
    f = fits[(fits.series == series) & (fits.strategy == "available") & (fits.status == "ok")
             & (fits.model != "pmc_state_gice")]
    f = f[np.isfinite(f.bic)]
    f = f[[not is_degenerate(r, args) for _, r in f.iterrows()]]
    best = f.loc[f.groupby(["model", "K"]).ll.idxmax()]
    out = []
    for pool in (best[best.model == "hmc_in"], best[best.model != "hmc_in"]):
        if len(pool):
            r = pool.loc[pool.bic.idxmin()]
            out.append({"model": r.model, "K": int(r.K), "start": int(r.start), "bic": float(r.bic)})
    return out


def phase2_specs(quick: bool, fits: pd.DataFrame, args) -> list[dict]:
    out = args.out
    specs = []
    stations = ("aotizhongxin",) if quick else ("huairou", "aotizhongxin")
    rates, blocks, seeds = ((0.10,), (6,), 1) if quick else (CV_RATES, CV_BLOCKS, CV_SEEDS)
    for st in stations:
        chosen = choose_models(fits, st, args)
        for rate in rates:
            for block in blocks:
                for seed in range(seeds):
                    specs.append({"task": "cv", "what": "baselines", "station": st, "rate": rate,
                                  "block": block, "seed": seed})
                    for ch in chosen:
                        tag = rc.fit_tag({"series": st, "K": ch["K"], "model": ch["model"],
                                          "strategy": "available", "start": ch["start"]})
                        specs.append({"task": "cv", "what": "model", "station": st, "rate": rate,
                                      "block": block, "seed": seed, "model": ch["model"],
                                      "K": ch["K"],
                                      "reference_labels": str(out / "labels" / f"{tag}.npz")})
    ch = choose_models(fits, "aotizhongxin", args)
    specs.append({"task": "forecast", "station": "aotizhongxin",
                  "models": [(c["model"], c["K"]) for c in ch],
                  "train_frac": 0.75, "h": 6 if quick else 24, "step": 200 if quick else 24})
    return specs


# ---------------------------------------------------------------------------
# Execution with a per-task cache
# ---------------------------------------------------------------------------

_VOLATILE = ("data_dir", "quick", "out")


def task_id(spec: dict) -> str:
    key = {k: v for k, v in spec.items() if k not in _VOLATILE}
    txt = json.dumps(key, sort_keys=True, default=str)
    head = spec["task"]
    if spec["task"] == "fit":
        head = rc.fit_tag(spec)
    elif spec["task"] == "cv":
        head = f"cv__{spec['station']}__{spec['what']}__{spec.get('model', '')}__r{spec['rate']}__b{spec['block']}__s{spec['seed']}"
    return f"{head}__{hashlib.sha1(txt.encode()).hexdigest()[:8]}"


def run_specs(specs: list[dict], args, label: str) -> list[tuple[dict, dict, float]]:
    cache = args.out / "tasks"
    cache.mkdir(parents=True, exist_ok=True)
    for sub in ("models", "labels", "imputations"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    done, todo = [], []
    for s in specs:
        s = {**s, "data_dir": args.data, "quick": args.quick, "out": str(args.out)}
        p = cache / f"{task_id(s)}.pkl"
        if p.exists() and not args.force:
            with open(p, "rb") as fh:
                done.append(pickle.load(fh))
        else:
            todo.append(s)
    todo.sort(key=_cost, reverse=True)
    print(f"[{label}] {len(specs)} tasks, {len(done)} cached, {len(todo)} to run "
          f"with {args.jobs} process(es)", flush=True)
    t0 = time.perf_counter()

    def _store(item):
        spec, res, wall = item
        with open(cache / f"{task_id(spec)}.pkl", "wb") as fh:
            pickle.dump(item, fh)
        done.append(item)
        n = len(done)
        if n % 10 == 0 or n == len(specs):
            print(f"[{label}] {n}/{len(specs)} done, {time.perf_counter() - t0:.0f} s", flush=True)
        for r in res.get("fits", []):
            if r["status"] != "ok":
                print(f"  FAILED {rc.fit_tag(spec)}: {r['error']}", flush=True)
        for r in res.get("task_errors", []):
            print(f"  TASK ERROR {r['task']}: {r['error']}", flush=True)

    if args.jobs <= 1:
        for s in todo:
            _store(rc.run_task(s))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(rc.run_task, s) for s in todo]
            for f in as_completed(futs):
                _store(f.result())
    # completion order depends on the process count: sort, so that the CSVs do not
    return sorted(done, key=lambda item: task_id(item[0]))


def collect(items, key: str) -> pd.DataFrame:
    rows = []
    for spec, res, wall in items:
        rows.extend(res.get(key, []))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def add_stability(fits: pd.DataFrame, args) -> pd.DataFrame:
    """LL gap and label agreement of every fit with the best fit of its cell."""
    fits = fits.copy()
    fits["ll_gap_to_best"] = np.nan
    fits["agree_with_best"] = np.nan
    fits["min_margin_sd"] = np.nan
    fits["y_sd"] = np.nan
    for i, r in fits[fits.status == "ok"].iterrows():
        raw = json.loads((args.out / "models" / f"{rc.fit_tag(r)}.json").read_text())
        fits.loc[i, "min_margin_sd"] = rc.min_margin_sd(rc.PMCModel.from_dict(raw))
        fits.loc[i, "y_sd"] = float(np.nanstd(rc.series_data(r.series, args.data, args.quick)))
        lls = np.load(args.out / "labels" / f"{rc.fit_tag(r)}.npz")["lls"]
        fits.loc[i, "ll_trace_max"] = float(np.max(lls)) if lls.size else np.nan
    fits["min_pi"] = fits.pi_sorted.map(lambda s: min(float(v) for v in str(s).split("|")) if isinstance(s, str) else np.nan)
    fits["degenerate"] = (fits.min_margin_sd < rc.DEGENERATE_SD_RATIO * fits.y_sd) | (fits.min_pi < rc.DEGENERATE_MIN_PI)
    ok = fits[fits.status == "ok"]
    for (series, K, model), g in ok.groupby(["series", "K", "model"]):
        b = g.loc[g.ll.idxmax()]
        ref = np.load(args.out / "labels" / f"{rc.fit_tag(b)}.npz")["x_hat"].astype(int)
        for i, r in g.iterrows():
            lab = np.load(args.out / "labels" / f"{rc.fit_tag(r)}.npz")["x_hat"].astype(int)
            fits.loc[i, "ll_gap_to_best"] = float(b.ll - r.ll)
            fits.loc[i, "agree_with_best"] = rc.agreement(ref, lab, int(K))
    return fits


def tsnh4_zoom(fits: pd.DataFrame, args) -> pd.DataFrame:
    """Around the longest real gap: truth, best-BIC model posterior, AR(1), linear."""
    d = rc.load_tsnh4(args.data, args.quick)
    Y, Yt = d["Y"], d["Y_true"]
    gl = rc.gap_lengths(~np.isfinite(Y))
    centre = int(np.argmax(gl))
    lo, hi = max(0, centre - 200), min(len(Y), centre + gl.max() + 200)
    f = fits[(fits.series == "tsnh4") & (fits.strategy == "available") & (fits.status == "ok") & ~fits.degenerate]
    b = f.loc[f.bic.idxmin()]
    imp = np.load(args.out / "imputations" / f"{rc.fit_tag(b)}.npz")
    bl = np.load(args.out / "imputations" / "tsnh4_baselines.npz")
    df = pd.DataFrame({"t": np.arange(lo, hi), "datetime": d["datetime"][lo:hi],
                       "y_obs": Y[lo:hi], "y_true": Yt[lo:hi]})
    for col in ("pmc_mean", "pmc_q05", "pmc_q95", "ar1_mean", "ar1_lo", "ar1_hi", "linear"):
        df[col] = np.nan
    pos = {int(t): k for k, t in enumerate(imp["index"])}
    for t in range(lo, hi):
        if t in pos:
            k = pos[t]
            df.loc[t - lo, ["pmc_mean", "pmc_q05", "pmc_q95"]] = (
                imp["mean"][k], imp["q"][k, 0], imp["q"][k, 2])
            df.loc[t - lo, ["ar1_mean", "ar1_lo", "ar1_hi", "linear"]] = (
                bl["ar1_mean"][k], bl["ar1_mean"][k] - rc.Z90 * bl["ar1_sd"][k],
                bl["ar1_mean"][k] + rc.Z90 * bl["ar1_sd"][k], bl["linear"][k])
    df["pmc_model"] = f"{b.model} K={b.K} start={b.start}"
    return df


def pamap2_strip(fits: pd.DataFrame, args) -> pd.DataFrame:
    """15 min of subject 102 (105 in --quick): labels, gaps and MPM labels (best K=3 fits)."""
    subj = 105 if args.quick else 102
    K = 2 if args.quick else 3
    st = np.load(args.out / f"pamap2_{subj}_strip.npz")
    groups = st["g3"] if K == 3 else rc.prep_pamap2(args.data, subj, args.quick)["g2"]
    keep = groups >= 0
    f = fits[(fits.status == "ok") & (fits.K == K) & ~fits.degenerate]
    comp = f[f.series == f"pamap2_{subj}_complete"]
    b = comp.loc[comp.bic.idxmin()]
    gap = f[(f.series == f"pamap2_{subj}_gapped") & (f.model == b.model) & (f.strategy == "available")]
    bg = gap.loc[gap.ll.idxmax()]
    out = {}
    for name, r in (("x_complete", b), ("x_gapped", bg)):
        lab = np.load(args.out / "labels" / f"{rc.fit_tag(r)}.npz")["x_hat"].astype(int)
        C = np.zeros((K, K), int)
        np.add.at(C, (groups[keep], lab[keep]), 1)
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(-C)
        relabel = np.arange(K)
        relabel[cols] = rows
        out[name] = relabel[lab]
    n = len(groups)
    width = min(n, 1800)
    # 15 min showing the most active groups (≥ 60 windows each; no 15 min contains all three
    # groups: rest only occurs at the start of the protocol) with the fewest transient windows
    best, lo = None, 0
    for start in range(0, n - width + 1, 50):
        g = groups[start: start + width]
        score = (sum(int((g == k).sum() >= 60) for k in range(1, K)), -int((g < 0).sum()))
        if best is None or score > best:
            best, lo = score, start
    hi = lo + width
    return pd.DataFrame({"t_s": st["t_s"][lo:hi], "Y": st["Y"][lo:hi], "group": groups[lo:hi],
                         "real": st["real"][lo:hi], "mcar": st["mcar"][lo:hi],
                         "x_complete": out["x_complete"][lo:hi], "x_gapped": out["x_gapped"][lo:hi],
                         "model": f"{b.model} K={K}"})


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=rc.DEFAULT_DATA, help="data folder (read-only)")
    ap.add_argument("--out", default=None, help="large outputs (default report/out/real_series/{full,quick})")
    ap.add_argument("--results", default=None,
                    help="small CSVs (default report/real_series/results; quick: <out>/results)")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="smoke run on short extracts (< 1 min)")
    ap.add_argument("--force", action="store_true", help="ignore the task cache")
    args = ap.parse_args(argv)
    args.out = Path(args.out) if args.out else REPO / "report/out/real_series" / ("quick" if args.quick else "full")
    results = Path(args.results) if args.results else (
        args.out / "results" if args.quick else HERE / "results")
    results.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    walls = {}

    t = time.perf_counter()
    items1 = run_specs(phase1_specs(args.quick), args, "phase 1")
    walls["phase1_s"] = time.perf_counter() - t
    fits = collect(items1, "fits")

    t = time.perf_counter()
    items2 = run_specs(phase2_specs(args.quick, fits, args), args, "phase 2")
    walls["phase2_s"] = time.perf_counter() - t

    t = time.perf_counter()
    items = items1 + items2
    fits = add_stability(fits, args)
    fits.sort_values(["series", "K", "model", "strategy", "start"]).to_csv(results / "fits.csv", index=False)
    for key, name in (("imputation", "imputation_tsnh4"), ("classification", "classification_pamap2"),
                      ("oracle", "oracle_pamap2"), ("mnar", "mnar_pamap2"),
                      ("pamap2_missingness", "missingness_pamap2"), ("cv", "cv_beijing"),
                      ("forecast", "forecast_beijing"), ("feature_check", "pamap2_feature_check"),
                      ("task_errors", "task_errors")):
        df = collect(items, key)
        if len(df):
            sort = [c for c in ("series", "subject", "station", "K", "model", "strategy", "start",
                                "rate", "block", "seed", "method", "gap_class", "h", "level", "code",
                                "rule", "applied_to") if c in df.columns]
            (df.sort_values(sort) if sort else df).to_csv(results / f"{name}.csv", index=False)
    # Figure extracts hold observations (tsNH4 is GPL-3): written outside the
    # repository, next to the other large outputs, never to ``results``.
    (args.out / "figure_data").mkdir(parents=True, exist_ok=True)
    for fn, name in ((tsnh4_zoom, "tsnh4_longest_gap"), (pamap2_strip, "pamap2_strip")):
        try:
            fn(fits, args).to_csv(args.out / "figure_data" / f"{name}.csv", index=False)
        except Exception as exc:
            print(f"  could not build {name}: {type(exc).__name__}: {exc}", flush=True)
    tasks = pd.DataFrame([{"task_id": task_id(s), "task": s["task"], "wall_s": w} for s, _, w in items])
    tasks.sort_values("task_id").to_csv(results / "tasks.csv", index=False)
    walls["phase3_s"] = time.perf_counter() - t
    walls["total_s"] = time.perf_counter() - t_start

    import pmcprg
    info = {"date": time.strftime("%Y-%m-%d %H:%M"), "quick": args.quick, "jobs": args.jobs,
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "pmcprg": getattr(pmcprg, "__version__", "?"), "machine": platform.machine(),
            "data": args.data, "this_invocation_wall_s": walls,
            "sum_task_wall_s": float(tasks.wall_s.sum()),
            "settings": {k: getattr(rc, k) for k in (
                "COPULA_CANDIDATES", "GICE_CANDIDATES", "GICE_RULE", "START_JITTER", "START_SEED",
                "ICE_MAX_ITER", "ICE_TOL", "SEM_MAX_ITER", "IMPUTE_DRAWS", "PAMAP2_WINDOW",
                "PAMAP2_MIN_MISSING", "PAMAP2_MCAR_RATE", "PAMAP2_MCAR_BLOCK", "DEQUANT_SEED")},
            "cv": {"rates": CV_RATES, "blocks": CV_BLOCKS, "seeds": CV_SEEDS}}
    (results / "run_info.json").write_text(json.dumps(info, indent=1, default=str))
    print(f"done in {walls['total_s']:.0f} s; results in {results}", flush=True)


if __name__ == "__main__":
    main()
