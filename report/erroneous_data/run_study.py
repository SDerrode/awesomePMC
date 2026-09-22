"""
run_study.py — isolated spikes in PMC/HMC series: PIT flags and flag-and-mask ICE.

Erroneous-data pilot (:mod:`pmcprg.pmc.outliers`). For every model ×
spike size k × contamination rate × replicate, a sequence (X, Y) of length
N is simulated from a fixture, isolated spikes y_n ← y_n + k·sd are added
(sd the standard deviation of the clean series; no two spikes adjacent,
never the first or last row), and:

Detection with the TRUE model
    ``flag_outliers`` at α = 1e-3 (no correction), sequential (innovation
    gating) and not: detection rate on the spikes, false-flag rate on the
    clean rows, and the false flags on the row right after a spike
    (``nbr``, the swamping that gating prevents).

Estimation (ICE from the true model, ``fit_margins``, Gaussian copulas
only for the PMC fixture, the package defaults otherwise)
    ``clean``   ICE on the clean series (the spikes never happened);
    ``raw``     ICE on the contaminated series;
    ``oracle``  ICE with the true spike rows set to NaN (the best a mask can
                do: its cost is the information of the masked rows);
    ``fm``      ``robust_estimate`` — fit, flag, mask, refit from the
                initial model until the mask is a fixed point;
    ``fm_warm`` the same loop with each refit started from the previous fit
                (the design ``robust_estimate`` rejected — measured here);
    ``fm_hampel`` ``robust_estimate`` with ``initial_mask`` = a Hampel
                pre-screen (|y_n − median| > 4 · 1.4826 · MAD over a
                centred window of 11 rows);
    ``dpd``     (PMC fixture) τ of the diagonal copulas re-estimated by
                minimum density power divergence (α_DPD = 0.25,
                :func:`pmcprg.copulas._robust.dpd_fit`) on the pseudo-pairs
                (F_i(y_n), F_i(y_{n+1})) of the raw fit, weights ξ_n(i, i) of
                the raw fit — the copula-level robust baseline (margins and
                prior stay those of ``raw``).

Classification error at the clean rows (Hungarian alignment)
    each fit classifying the series it was fitted on (the masked rows
    integrated out), plus the true model on the raw series (``true_raw``)
    and on the series with its sequential flags masked (``true_seq``).

Seeds are ``zlib.crc32`` of labels: the clean series depends on (model, rep)
only — every (k, rate) cell of a replicate contaminates the same series —
and the spike positions on (model, k, rate, rep). ``--jobs`` changes no
number.

Usage (from the repository root)
--------------------------------
    PYTHONPATH=. .venv/bin/python report/erroneous_data/run_study.py --jobs 4
    PYTHONPATH=. .venv/bin/python report/erroneous_data/run_study.py --summarise

writes ``report/erroneous_data/results/runs.csv`` (one row per series ×
method), ``run_info.json`` and ``tables.md`` (the tables of the README).
``--quick``: 2 replicates, N = 1000, into ``results/quick/`` (not versioned).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import platform
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODELS = {"hmc_in_gauss_k2": "HMC-IN", "pmc_gauss_k2": "PMC"}
KS = (4, 6, 8)
RATES = (0.01, 0.05)
ALPHA = 1e-3
DPD_ALPHA = 0.25
HAMPEL_HALF, HAMPEL_T = 5, 4.0
PARAMS = ("mu0", "mu1", "sd0", "sd1", "A00", "A11", "tau00", "tau11")


def _seed(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode())


def _spikes(N: int, rate: float, rng: np.random.Generator) -> np.ndarray:
    """round(rate·N) isolated positions in 1..N−2, no two adjacent."""
    want = int(round(rate * N))
    chosen = np.zeros(N, dtype=bool)
    k = 0
    for c in rng.permutation(np.arange(1, N - 1)):
        if k == want:
            break
        if chosen[c - 1] or chosen[c + 1]:
            continue
        chosen[c] = True
        k += 1
    return chosen


def hampel(y: np.ndarray, half: int = HAMPEL_HALF, t: float = HAMPEL_T) -> np.ndarray:
    """Hampel identifier: |y_n − med| > t · 1.4826 · MAD on a centred window."""
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.pad(y, half, mode="reflect")
    win = sliding_window_view(pad, 2 * half + 1)
    med = np.median(win, axis=1)
    mad = 1.4826 * np.median(np.abs(win - med[:, None]), axis=1)
    return np.abs(y - med) > t * np.maximum(mad, 1e-12)


def _params(model) -> dict:
    blocks = model.margin_blocks()
    p = model.prior_p
    A = p / p.sum(axis=1, keepdims=True)
    out = {"mu0": blocks[0]["params"]["loc"], "mu1": blocks[1]["params"]["loc"],
           "sd0": blocks[0]["params"]["scale"], "sd1": blocks[1]["params"]["scale"],
           "A00": A[0, 0], "A11": A[1, 1], "tau00": math.nan, "tau11": math.nan}
    if model.variant.uses_copula:
        for blk in model.copula_blocks():
            if int(blk["i"]) == int(blk["j"]) < 2:
                out[f"tau{blk['i']}{blk['i']}"] = float(blk["tau"])
    return {k: float(v) for k, v in out.items()}


def _clean_err(model_fit, Yfit, X, clean) -> float:
    from pmcprg.pmc import classify, error_rate
    Xh = classify(model_fit, Yfit)[0]
    return float(error_rate(X[clean], Xh[clean]))


def _warm_loop(model, Yc, cfg, max_rounds=10):
    """robust_estimate's loop with each refit started from the previous fit."""
    from pmcprg.pmc import flag_outliers, ice
    mask = np.zeros(len(Yc), dtype=bool)
    fitted, _ = ice(model, Yc, cfg)
    fits = 1
    for _ in range(max_rounds):
        f = flag_outliers(fitted, Yc, alpha=ALPHA)
        if np.array_equal(f.flagged, mask):
            return fitted, mask, fits, True
        mask = f.flagged.copy()
        Ym = Yc.copy()
        Ym[mask] = np.nan
        fitted, _ = ice(fitted, Ym, {**cfg, "init": "model"})
        fits += 1
    f = flag_outliers(fitted, Yc, alpha=ALPHA)
    return fitted, mask, fits, bool(np.array_equal(f.flagged, mask))


def _dpd_taus(raw_fit, Yc) -> dict:
    from pmcprg.copulas._robust import dpd_fit
    from pmcprg.copulas.elliptical.gaussian import CopulaGaussian
    from pmcprg.pmc import gap_posterior
    xi = gap_posterior(raw_fit, Yc).xi
    out = {}
    for i in range(raw_fit.K):
        F = raw_fit.margin(i).cdf_vec(Yc)
        uv = np.clip(np.column_stack((F[:-1], F[1:])), 1e-12, 1 - 1e-12)
        out[f"tau{i}{i}"] = float(dpd_fit(CopulaGaussian, uv, DPD_ALPHA, weights=xi[:, i, i]).tau_k)
    return out


def run_series(task: tuple) -> list[dict]:
    logging.disable(logging.WARNING)
    from pmcprg.pmc import PMCModel, flag_outliers, ice, robust_estimate, simulate

    name, k, rate, rep, N = task
    model = PMCModel(ROOT / "pmcprg" / "pmc" / "models" / f"{name}.toml")
    X, Y = simulate(model, N, seed=_seed("sim", name, rep))
    cont = (_spikes(N, rate, np.random.default_rng(_seed("cont", name, k, rate, rep)))
            if rate > 0 else np.zeros(N, dtype=bool))
    Yc = Y.copy()
    Yc[cont] += k * Y.std()
    clean = ~cont
    nbr = np.zeros(N, dtype=bool)
    nbr[np.nonzero(cont)[0] + 1] = True
    nbr &= clean
    cfg = {"fit_margins": True}
    if model.variant.uses_copula:
        cfg["candidates"] = ["Gauss"]
    base = dict(model=name, k=k, rate=rate, rep=rep, N=N, n_spikes=int(cont.sum()))
    rows = []

    def det(mask):
        return dict(detect=float(mask[cont].mean()) if cont.any() else math.nan,
                    false_rate=float(mask[clean].mean()),
                    nbr_false=int((mask & nbr).sum()), n_masked=int(mask.sum()))

    # Detection with the true model.
    for seq in (True, False):
        t0 = time.perf_counter()
        f = flag_outliers(model, Yc, alpha=ALPHA, sequential=seq)
        rows.append({**base, "method": "true_seq" if seq else "true_nonseq", **det(f.flagged),
                     "err_clean": math.nan, "fits": 0, "converged": True,
                     "seconds": time.perf_counter() - t0})
        if seq:
            Ytm = Yc.copy()
            Ytm[f.flagged] = np.nan
            rows[-1]["err_clean"] = _clean_err(model, Ytm, X, clean)
    rows.append({**base, "method": "true_raw", **det(np.zeros(N, bool)),
                 "err_clean": _clean_err(model, Yc, X, clean), "fits": 0,
                 "converged": True, "seconds": 0.0})

    def add(method, fitted, Yfit, mask, fits, converged, t0, extra=None):
        rows.append({**base, "method": method, **det(mask),
                     "err_clean": _clean_err(fitted, Yfit, X, clean), "fits": fits,
                     "converged": converged, "seconds": time.perf_counter() - t0,
                     **_params(fitted), **(extra or {})})

    t0 = time.perf_counter()
    fit, _ = ice(model, Y, cfg)
    add("clean", fit, Y, np.zeros(N, bool), 1, True, t0)

    t0 = time.perf_counter()
    raw, _ = ice(model, Yc, cfg)
    add("raw", raw, Yc, np.zeros(N, bool), 1, True, t0)

    t0 = time.perf_counter()
    Yo = Yc.copy()
    Yo[cont] = np.nan
    fit, _ = ice(model, Yo, cfg)
    add("oracle", fit, Yo, cont, 1, True, t0)

    for method, init in (("fm", None), ("fm_hampel", hampel(Yc))):
        t0 = time.perf_counter()
        rf = robust_estimate(model, Yc, cfg, flag_cfg={"alpha": ALPHA}, initial_mask=init)
        Ym = Yc.copy()
        Ym[rf.mask] = np.nan
        extra = None
        if init is not None:
            extra = {"prescreen_detect": float(init[cont].mean()) if cont.any() else math.nan,
                     "prescreen_false": float(init[clean].mean())}
        add(method, rf.model, Ym, rf.mask, rf.n_fits, rf.converged, t0, extra)

    t0 = time.perf_counter()
    fitted, mask, fits, conv = _warm_loop(model, Yc, cfg)
    Ym = Yc.copy()
    Ym[mask] = np.nan
    add("fm_warm", fitted, Ym, mask, fits, conv, t0)

    if model.variant.uses_copula:
        t0 = time.perf_counter()
        p = _params(raw)
        p.update(_dpd_taus(raw, Yc))
        rows.append({**base, "method": "dpd", **det(np.zeros(N, bool)), "err_clean": math.nan,
                     "fits": 0, "converged": True, "seconds": time.perf_counter() - t0, **p})
    return rows


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def _truth(name: str) -> dict:
    from pmcprg.pmc import PMCModel
    return _params(PMCModel(ROOT / "pmcprg" / "pmc" / "models" / f"{name}.toml"))


def summarise(results: Path) -> str:
    rows = list(csv.DictReader(open(results / "runs.csv")))
    for r in rows:
        for key in ("k", "rep", "N", "n_spikes", "fits", "nbr_false", "n_masked"):
            r[key] = int(r[key])
        for key in ("rate", "detect", "false_rate", "err_clean", "seconds", *PARAMS,
                    "prescreen_detect", "prescreen_false"):
            v = r.get(key, "")
            r[key] = float(v) if v not in ("", None) else math.nan
        r["converged"] = r["converged"] == "True"

    def sel(**kw):
        return [r for r in rows if all(r[a] == b for a, b in kw.items())]

    def mean(vals):
        vals = [v for v in vals if not math.isnan(v)]
        return float(np.mean(vals)) if vals else math.nan

    info = json.load(open(results / "run_info.json"))
    out = [f"<!-- generated by run_study.py --summarise from {results.name}/runs.csv -->", ""]
    out.append(f"N = {info['N']}, {info['reps']} replicates per cell, α = {ALPHA:g}; "
               f"{info['n_series']} series, {info['wall_seconds'] / 60:.1f} min with "
               f"{info['jobs']} processes ({info['machine']}).")
    out.append("")
    cells = [(k, rate) for rate in RATES for k in KS]

    # 1. Detection with the true model.
    out += ["### Detection with the true model (flag_outliers, α = 1e-3)", "",
            "Detection rate on the spikes / false-flag rate on the clean rows "
            "(per 1000) / false flags right after a spike (total over the replicates).", ""]
    out.append("| model | k | rate | sequential | non-sequential |")
    out.append("|---|---|---|---|---|")
    for name in MODELS:
        for rate in (0.0, *RATES):
            for k in (KS if rate else (0,)):
                cell = []
                for meth in ("true_seq", "true_nonseq"):
                    rr = sel(model=name, k=k, rate=rate, method=meth)
                    cell.append(f"{mean([r['detect'] for r in rr]):.3f} / "
                                f"{1000 * mean([r['false_rate'] for r in rr]):.2f} / "
                                f"{sum(r['nbr_false'] for r in rr)}"
                                if rate else
                                f"– / {1000 * mean([r['false_rate'] for r in rr]):.2f} / –")
                out.append(f"| {MODELS[name]} | {k if rate else '–'} | {rate:.0%} | {cell[0]} | {cell[1]} |")
    out.append("")

    # 2. Parameter bias.
    meths = ("clean", "raw", "fm", "fm_hampel", "oracle")
    for name in MODELS:
        truth = _truth(name)
        keys = [p for p in PARAMS if not math.isnan(truth[p])]
        out += [f"### {MODELS[name]}: bias of the ICE estimates (mean over replicates of estimate − truth)", ""]
        out.append("| k | rate | method | " + " | ".join(keys) + " | fits | converged |")
        out.append("|---|---|---|" + "---|" * len(keys) + "---|---|")
        for rate in (0.0, *RATES):
            for k in (KS if rate else (0,)):
                for meth in (meths if rate else ("clean", "fm")) + (("dpd",) if rate and name == "pmc_gauss_k2" else ()):
                    rr = sel(model=name, k=k, rate=rate, method=meth)
                    if not rr:
                        continue
                    vals = []
                    for p in keys:
                        b = mean([r[p] for r in rr]) - truth[p]
                        vals.append("" if math.isnan(b) else f"{b:+.3f}")
                    fits = mean([r["fits"] for r in rr]) if meth.startswith("fm") else math.nan
                    conv = (f"{sum(r['converged'] for r in rr)}/{len(rr)}"
                            if meth.startswith("fm") else "")
                    out.append(f"| {k if rate else '–'} | {rate:.0%} | {meth} | " + " | ".join(vals)
                               + f" | {'' if math.isnan(fits) else f'{fits:.1f}'} | {conv} |")
        out.append("")

    # 3. RMSE of the key parameters.
    for name in MODELS:
        truth = _truth(name)
        keys = [p for p in PARAMS if not math.isnan(truth[p])]
        out += [f"### {MODELS[name]}: RMSE of the ICE estimates", ""]
        out.append("| k | rate | method | " + " | ".join(keys) + " |")
        out.append("|---|---|---|" + "---|" * len(keys))
        for k, rate in cells:
            for meth in meths + (("dpd",) if name == "pmc_gauss_k2" else ()):
                rr = sel(model=name, k=k, rate=rate, method=meth)
                vals = []
                for p in keys:
                    v = [r[p] - truth[p] for r in rr if not math.isnan(r[p])]
                    vals.append(f"{math.sqrt(np.mean(np.square(v))):.3f}" if v else "")
                out.append(f"| {k} | {rate:.0%} | {meth} | " + " | ".join(vals) + " |")
        out.append("")

    # 4. Classification at the clean rows, and the masks of the fitted procedures.
    out += ["### Classification error at the clean rows (%), and the masks of the fitted procedures", "",
            "Error: mean over replicates. Mask: detection rate / false-flag rate per 1000 clean rows.", ""]
    cm = ("true_raw", "true_seq", "clean", "raw", "fm", "fm_hampel", "fm_warm", "oracle")
    out.append("| model | k | rate | " + " | ".join(cm) + " | mask fm | mask fm_hampel | mask fm_warm |")
    out.append("|---|---|---|" + "---|" * (len(cm) + 3))
    for name in MODELS:
        for rate in (0.0, *RATES):
            for k in (KS if rate else (0,)):
                errs = [f"{100 * mean([r['err_clean'] for r in sel(model=name, k=k, rate=rate, method=m)]):.2f}"
                        for m in cm]
                masks = []
                for m in ("fm", "fm_hampel", "fm_warm"):
                    rr = sel(model=name, k=k, rate=rate, method=m)
                    d = mean([r["detect"] for r in rr])
                    masks.append(("–" if math.isnan(d) else f"{d:.3f}")
                                 + f" / {1000 * mean([r['false_rate'] for r in rr]):.2f}")
                out.append(f"| {MODELS[name]} | {k if rate else '–'} | {rate:.0%} | "
                           + " | ".join(errs) + " | " + " | ".join(masks) + " |")
    out.append("")

    # 5. Pre-screen and cost.
    out += ["### Hampel pre-screen and cost", "",
            "Pre-screen detection / false rate per 1000; mean wall time per series (s).", ""]
    out.append("| model | k | rate | pre-screen | raw s | fm s | fm_hampel s | fm_warm s |")
    out.append("|---|---|---|---|---|---|---|---|")
    for name in MODELS:
        for rate in (0.0, *RATES):
            for k in (KS if rate else (0,)):
                rr = sel(model=name, k=k, rate=rate, method="fm_hampel")
                d = mean([r["prescreen_detect"] for r in rr])
                pre = ("–" if math.isnan(d) else f"{d:.3f}") + \
                    f" / {1000 * mean([r['prescreen_false'] for r in rr]):.2f}"
                secs = [f"{mean([r['seconds'] for r in sel(model=name, k=k, rate=rate, method=m)]):.2f}"
                        for m in ("raw", "fm", "fm_hampel", "fm_warm")]
                out.append(f"| {MODELS[name]} | {k if rate else '–'} | {rate:.0%} | {pre} | "
                           + " | ".join(secs) + " |")
    out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--n-obs", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--summarise", action="store_true", help="tables only, no simulation")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (HERE / "results" / ("quick" if args.quick else ""))
    out.mkdir(parents=True, exist_ok=True)
    if args.summarise:
        (out / "tables.md").write_text(summarise(out))
        print(f"wrote {out / 'tables.md'}")
        return
    reps, N = (2, 1000) if args.quick else (args.reps, args.n_obs)
    tasks = [(name, k, rate, rep, N) for name in MODELS for rep in range(reps)
             for rate in (0.0, *RATES) for k in (KS if rate else (0,))]
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(args.jobs) as ex:
        for i, rr in enumerate(ex.map(run_series, tasks, chunksize=1)):
            rows += rr
            if (i + 1) % 10 == 0:
                print(f"{i + 1}/{len(tasks)} series, {time.perf_counter() - t0:.0f} s", flush=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    import numpy
    import scipy
    json.dump({"N": N, "reps": reps, "n_series": len(tasks), "jobs": args.jobs,
               "wall_seconds": time.perf_counter() - t0, "alpha": ALPHA,
               "machine": f"{platform.system()} {platform.machine()}",
               "python": platform.python_version(), "numpy": numpy.__version__,
               "scipy": scipy.__version__, "omp_num_threads": os.environ.get("OMP_NUM_THREADS")},
              open(out / "run_info.json", "w"), indent=2)
    (out / "tables.md").write_text(summarise(out))
    print(f"{len(tasks)} series in {time.perf_counter() - t0:.0f} s → {out}")


if __name__ == "__main__":
    main()
