"""Monte-Carlo study of ``missingness_lr_test`` and of the mechanism estimates (P6).

Size and power of the likelihood-ratio test of state-dependent missingness
(:func:`pmcprg.pmc.missingness_lr.missingness_lr_test`), asymptotic χ² and
parametric bootstrap, and recovery (bias, RMSE) of π̂, â, b̂ fitted by ICE.

Design
------
Models (K = 2, ICE ``fit_margins = True``, started from the true θ without
mechanism, the test's ICE defaults — ``tol = 1e-8``, ``patience = 500``):

* ``hmc_in`` — HMC-IN, A = [[0.95, 0.05], [0.05, 0.95]], margins N(∓1, 1):
  the exact K-state shortcut, where ICE is EM;
* ``hmc_dn`` — ``pmcprg/pmc/models/hmc_dn_gauss_k2.toml`` (HMC-DN, A = 0.9 on
  the diagonal, margins N(∓1, 1), Gaussian copulas τ = 0.6 / 0): the
  quadrature grid (64 nodes), ICE with ``candidates = ["Gauss"]``.

Scenarios (``SCENARIOS``): ``"state"`` masks with π = (0.1, 0.1) (H0) and
π = 0.1 ∓ δ/2 for δ = 0.05, 0.10, 0.15; ``"state-markov"`` masks with a
common (a, b) = (0.02, 0.8) (H0: 9 % missing, bursts of mean length 5) and
two alternatives, onset only (a = (0.013, 0.027), b = 0.8) and onset and
persistence (a = (0.01, 0.03), b = (0.7, 0.9)). "state" data are tested
"state" vs common, "state-markov" data "state-markov" vs common; on HMC-IN
the nested "state-markov" vs "state" test runs too (H0: π = (0.05, 0.15)).

Bootstrap. The size and power of the bootstrap test come from the warp-speed
method (Giacomini, Politis & White 2013): each replication r draws ONE
bootstrap replicate LR*_r from its own fitted null model
(:func:`~pmcprg.pmc.missingness_lr.bootstrap_replicate`), and the test
rejects when LR_r exceeds the 95 % quantile of {LR*_r} pooled over the
replications of the cell — the cost of 2 fits per replication instead of 2B.
``--direct`` checks it on one cell with a genuine B = 99 bootstrap per
replication.

Seeds: ``zlib.crc32`` of named tuples; the path, the mask and the bootstrap
path and mask of a replication have four distinct seeds.

The statistic. ``results/lr_study.csv`` and ``results/lr_direct.csv`` were
measured when the statistic was the difference of the two ICE fits,
2 (LL1(θ̂1, φ̂1) − LL0(θ̂0)) — now ``statistic_fits`` (column ``LR`` there).
``--profile`` reruns the cells of ``PROFILE_PLAN`` with the same seeds and
the profile statistic of :mod:`pmcprg.pmc.missingness_lr` (column ``LR``;
``LR_fits`` is the fits' difference, bit-identical to the first study's
``LR``), for a paired comparison (``summarise_profile``). ``--direct-study``
runs a genuine B = 99 bootstrap on replications of the HMC-DN Markov null of
the study (``DIRECT_STUDY_R``). Why: ``lr_diagnosis.py`` and ``README.md``,
"Diagnosis".

Run (from the repository root)::

    .venv/bin/python report/missing_state/lr_study.py --jobs 6            # everything
    .venv/bin/python report/missing_state/lr_study.py --jobs 6 --direct   # warp-speed check
    .venv/bin/python report/missing_state/lr_study.py --jobs 6 --profile  # paired rerun
    .venv/bin/python report/missing_state/lr_study.py --jobs 6 --direct-study
    .venv/bin/python report/missing_state/lr_study.py --summarise         # tables only

Writes ``results/lr_study.csv`` (one row per replication),
``results/lr_direct.csv``, ``results/lr_study_profile.csv`` and
``results/lr_direct_dn.csv``, and prints the tables of ``README.md``.

References
----------
* Giacomini, R., Politis, D. N. & White, H. (2013). A warp-speed method for
  conducting Monte Carlo experiments involving bootstrap estimators.
  *Econometric Theory* 29(3), 567–589.
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = Path(__file__).resolve().parent / "results"

ICE_CFG = {
    "hmc_in": {"fit_margins": True},
    "hmc_dn": {"fit_margins": True, "candidates": ["Gauss"]},
}

#: name -> (mechanism, params, test (alternative, null), role)
SCENARIOS = {
    "state-null":      ("state", {"rates": (0.1, 0.1)}, ("state", "common"), "size"),
    "state-d0.05":     ("state", {"rates": (0.075, 0.125)}, ("state", "common"), "power"),
    "state-d0.10":     ("state", {"rates": (0.05, 0.15)}, ("state", "common"), "power"),
    "state-d0.15":     ("state", {"rates": (0.025, 0.175)}, ("state", "common"), "power"),
    "markov-null":     ("state-markov", {"onset": (0.02, 0.02), "persistence": (0.8, 0.8)},
                        ("state-markov", "common"), "size"),
    "markov-onset":    ("state-markov", {"onset": (0.013, 0.027), "persistence": (0.8, 0.8)},
                        ("state-markov", "common"), "power"),
    "markov-both":     ("state-markov", {"onset": (0.01, 0.03), "persistence": (0.7, 0.9)},
                        ("state-markov", "common"), "power"),
    "nested-null":     ("state", {"rates": (0.05, 0.15)}, ("state-markov", "state"), "size"),
    "nested-both":     ("state-markov", {"onset": (0.01, 0.03), "persistence": (0.7, 0.9)},
                        ("state-markov", "state"), "power"),
}

#: (model, scenario) -> (Ns, R)
PLAN = {
    **{("hmc_in", s): ((500, 1000, 2000), 400 if SCENARIOS[s][3] == "size" else 200)
       for s in SCENARIOS},
    **{("hmc_dn", s): ((500, 1000), 200 if SCENARIOS[s][3] == "size" else 100)
       for s in ("state-null", "state-d0.10", "state-d0.15", "markov-null", "markov-both")},
    # recovery at a larger N (no bootstrap replicate)
    ("hmc_in", "state-d0.10@5000"): ((5000,), 100),
    ("hmc_in", "markov-both@5000"): ((5000,), 100),
}

#: The cells rerun with the profile statistic (``--profile``): HMC-IN at
#: N = 500, every scenario (the three tests), and the HMC-DN Markov test at
#: N = 500 and 1000 — same seeds as ``PLAN``, so paired with ``lr_study.csv``.
PROFILE_PLAN = {
    **{("hmc_in", s): ((500,), R) for (m, s), (_, R) in PLAN.items()
       if m == "hmc_in" and "@" not in s},
    **{("hmc_dn", s): ((500, 1000), R) for (m, s), (_, R) in PLAN.items()
       if m == "hmc_dn" and s.startswith("markov")},
}

#: Replications of the HMC-DN Markov null (N = 500) given a genuine B = 99
#: bootstrap (``--direct-study``): the four largest statistics of the study
#: (LR of the fits 11.1–16.6; 10.6–14.0 by direct maximisation) and the
#: first two.
DIRECT_STUDY_R = (78, 197, 11, 198, 0, 1)


def _seed(*parts) -> int:
    return zlib.crc32(repr(parts).encode()) % 2**31


def _model(name):
    from pmcprg.pmc import PMCModel
    if name == "hmc_in":
        return PMCModel.from_dict({
            "model": {"variant": "HMC-IN", "K": 2},
            "prior": {"A": [[0.95, 0.05], [0.05, 0.95]]},
            "margins": [{"i": i, "dist": "norm", "params": {"loc": m, "scale": 1.0}}
                        for i, m in enumerate((-1.0, 1.0))]})
    return PMCModel(REPO / "pmcprg" / "pmc" / "models" / "hmc_dn_gauss_k2.toml")


def _mask(Y, X, mech, params, seed):
    from pmcprg.missing.patterns import state_dependent, state_markov
    if mech == "state":
        return state_dependent(Y, X, params["rates"], seed=seed)[0]
    return state_markov(Y, X, params["onset"], params["persistence"], seed=seed)[0]


def _flat(prefix, table):
    out = {}
    for k in ("rates", "onset", "persistence"):
        for i, v in enumerate(table.get(k, ())):
            out[f"{prefix}{k}{i}"] = float(v)
    return out


def _job(args):
    model_name, scen_key, N, r, warp = args
    import logging as _lg
    _lg.disable(_lg.WARNING)
    from pmcprg.pmc import simulate
    from pmcprg.pmc.missingness_lr import bootstrap_replicate, missingness_lr_test
    scen = scen_key.split("@")[0]
    mech, params, (alt, null), role = SCENARIOS[scen]
    m = _model(model_name)
    t0 = time.time()
    X, Y = simulate(m, N=N, seed=_seed("sim", model_name, scen_key, N, r))
    Yn = _mask(Y, X, mech, params, _seed("mask", model_name, scen_key, N, r))
    cfg = ICE_CFG[model_name]
    row = {"model": model_name, "scenario": scen_key, "role": role, "alternative": alt,
           "null": null, "N": N, "r": r, "M": int(np.isnan(Yn).sum())}
    try:
        res = missingness_lr_test(m, Yn, alternative=alt, null=null, ice_cfg=cfg)
    except ValueError as exc:                     # empty mask (never at these rates)
        row["error"] = str(exc)
        return row
    row.update(LR=res.statistic, df=res.df, p_asym=res.p_value,
               ll0=res.log_lik_null, ll1=res.log_lik_alt,
               LR_fits=res.statistic_fits, sup0=res.sup_log_lik_null, sup1=res.sup_log_lik_alt,
               **{f"prof_{k}": v for k, v in res.profile_log_liks.items()},
               **_flat("true_", params), **_flat("alt_", res.alt_params),
               **_flat("null_", res.null_params))
    if warp:
        lr_b = bootstrap_replicate(res.null_model, N,
                                   _seed("warp-sim", model_name, scen_key, N, r),
                                   _seed("warp-mask", model_name, scen_key, N, r),
                                   alternative=alt, null=null, ice_cfg=cfg)
        row["LR_warp"] = np.nan if lr_b is None else lr_b
    row["seconds"] = time.time() - t0
    return row


def _job_direct(args):
    """A genuine B-replicate bootstrap. ``tag`` "direct": data of their own
    (seeds "direct-sim", "direct-mask"); "study": the data of replication r
    of the study (seeds "sim", "mask" — the same series and mask as
    ``_job``)."""
    model_name, scen, N, r, B, tag = args
    import logging as _lg
    _lg.disable(_lg.WARNING)
    from pmcprg.pmc import simulate
    from pmcprg.pmc.missingness_lr import missingness_lr_test
    mech, params, (alt, null), role = SCENARIOS[scen]
    m = _model(model_name)
    t0 = time.time()
    pre = "direct-" if tag == "direct" else ""
    X, Y = simulate(m, N=N, seed=_seed(f"{pre}sim", model_name, scen, N, r))
    Yn = _mask(Y, X, mech, params, _seed(f"{pre}mask", model_name, scen, N, r))
    res = missingness_lr_test(m, Yn, alternative=alt, null=null, ice_cfg=ICE_CFG[model_name],
                              n_bootstrap=B, seed=_seed("direct-boot", model_name, scen, N, r))
    boot = np.asarray(res.bootstrap_statistics, dtype=float)
    return {"model": model_name, "scenario": scen, "N": N, "r": r, "B": B, "data": tag,
            "LR": res.statistic, "LR_fits": res.statistic_fits, "p_asym": res.p_value,
            "p_boot": res.p_value_bootstrap, "B_valid": res.n_bootstrap_valid,
            "boot_q95": float(np.quantile(boot, 0.95)) if boot.size else np.nan,
            "boot_mean": float(boot.mean()) if boot.size else np.nan,
            "seconds": time.time() - t0}


def _run(fn, jobs, workers, path):
    OUT.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(fn, j) for j in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if k % 200 == 0 or k == len(futs):
                print(f"  {k}/{len(futs)} ({time.time() - t0:.0f} s)", flush=True)
    rows.sort(key=lambda d: tuple(str(d.get(k)) for k in ("model", "scenario", "N"))
              + (int(d["r"]),))
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return rows


def _read(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v) if v not in ("", None) else np.nan
            except ValueError:
                pass
    return rows


def _rate(x):
    x = np.asarray(x, dtype=float)
    p = float(x.mean())
    return p, math.sqrt(p * (1 - p) / x.size)


def summarise(path=OUT / "lr_study.csv", direct=OUT / "lr_direct.csv"):
    rows = [r for r in _read(path) if not isinstance(r.get("error"), str)]
    cells = {}
    for r in rows:
        cells.setdefault((r["model"], r["scenario"], int(r["N"])), []).append(r)

    print("\n### Size and power at the 5 % level\n")
    print("| model | test (H1 vs H0) | scenario | N | R | mean M | mean LR (df) "
          "| reject χ² | reject bootstrap (warp-speed) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (mdl, scen, N), rs in sorted(cells.items(), key=lambda kv: (kv[0][0], list(SCENARIOS).index(kv[0][1].split("@")[0]), kv[0][2])):
        if "@" in scen:
            continue
        lr = np.array([r["LR"] for r in rs])
        p = np.array([r["p_asym"] for r in rs])
        warp = np.array([r.get("LR_warp", np.nan) for r in rs], dtype=float)
        a, sa = _rate(p < 0.05)
        wv = warp[np.isfinite(warp)]
        crit = np.quantile(wv, 0.95) if wv.size else np.nan
        b, sb = _rate(lr > crit)
        df = int(rs[0]["df"])
        print(f"| {mdl} | {rs[0]['alternative']} vs {rs[0]['null']} | {scen} | {N} | {len(rs)} | "
              f"{np.mean([r['M'] for r in rs]):.0f} | {lr.mean():.2f} ({df}) | "
              f"{a:.3f} ± {sa:.3f} | {b:.3f} ± {sb:.3f} (crit {crit:.2f}, χ² {_chi2q(df):.2f}) |")

    print("\n### Recovery of the mechanism (alternative fit)\n")
    print("| model | scenario | N | R | parameter | true | bias | RMSE | sd |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (mdl, scen, N), rs in sorted(cells.items(), key=lambda kv: (kv[0][0], list(SCENARIOS).index(kv[0][1].split("@")[0]), kv[0][2])):
        if scen.startswith("nested"):
            continue
        keys = sorted(k[5:] for k in rs[0] if k.startswith("true_"))
        for k in keys:
            t = rs[0][f"true_{k}"]
            if f"alt_{k}" not in rs[0] or not np.isfinite(t):
                continue
            x = np.array([r[f"alt_{k}"] for r in rs])
            print(f"| {mdl} | {scen} | {N} | {len(rs)} | {k[:-1]}[{k[-1]}] | {t:g} | "
                  f"{x.mean() - t:+.4f} | {math.sqrt(((x - t) ** 2).mean()):.4f} | "
                  f"{x.std(ddof=1):.4f} |")

    if Path(direct).exists():
        d = _read(direct)
        print("\n### Direct bootstrap (B per replication) vs warp-speed\n")
        print("| model | scenario | N | R | B | reject χ² | reject bootstrap | mean B valid |")
        print("|---|---|---|---|---|---|---|---|")
        groups = {}
        for r in d:
            groups.setdefault((r["model"], r["scenario"], int(r["N"]), int(r["B"])), []).append(r)
        for (mdl, scen, N, B), rs in sorted(groups.items()):
            a, sa = _rate(np.array([r["p_asym"] for r in rs]) < 0.05)
            b, sb = _rate(np.array([r["p_boot"] for r in rs]) <= 0.05)
            print(f"| {mdl} | {scen} | {N} | {len(rs)} | {B} | {a:.3f} ± {sa:.3f} | "
                  f"{b:.3f} ± {sb:.3f} | {np.mean([r['B_valid'] for r in rs]):.1f} |")


def summarise_profile(path=OUT / "lr_study_profile.csv", before=OUT / "lr_study.csv",
                      direct=OUT / "lr_direct_dn.csv"):
    """Paired table: the fits' statistic (``lr_study.csv``, same seeds) against
    the profile statistic (``lr_study_profile.csv``)."""
    if not Path(path).exists():
        return
    old = {(r["model"], r["scenario"], int(r["N"]), int(r["r"])): r for r in _read(before)}
    cells = {}
    for r in _read(path):
        cells.setdefault((r["model"], r["scenario"], int(r["N"])), []).append(r)
    print("\n### Before (difference of the fits) and after (profile statistic), paired\n")
    print("| model | test | scenario | N | R | fits: mean LR, #<0 | reject χ² | reject boot (crit) "
          "| profile: mean LR, min | reject χ² | reject boot (crit) | max \\|LR − fits\\| | fits bit-identical |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    order = lambda kv: (kv[0][0], list(SCENARIOS).index(kv[0][1]), kv[0][2])   # noqa: E731
    for (mdl, scen, N), rs in sorted(cells.items(), key=order):
        pre = [old[(mdl, scen, N, int(r["r"]))] for r in rs]
        df = int(rs[0]["df"])
        q = _chi2q(df)
        lr_old = np.array([p["LR"] for p in pre])
        lr_fit = np.array([r["LR_fits"] for r in rs])
        lr_new = np.array([r["LR"] for r in rs])
        same = int(np.sum(lr_old == lr_fit))

        def boot(lr, warp):
            w = np.asarray(warp, dtype=float)
            w = w[np.isfinite(w)]
            if not w.size:
                return "—"
            c = float(np.quantile(w, 0.95))
            b, sb = _rate(lr > c)
            return f"{b:.3f} ± {sb:.3f} ({c:.2f})"
        a0, s0 = _rate(lr_old > q)
        a1, s1 = _rate(lr_new > q)
        print(f"| {mdl} | {rs[0]['alternative']} vs {rs[0]['null']} | {scen} | {N} | {len(rs)} | "
              f"{lr_old.mean():.2f}, {int((lr_old < 0).sum())} | {a0:.3f} ± {s0:.3f} | "
              f"{boot(lr_old, [p.get('LR_warp', np.nan) for p in pre])} | "
              f"{lr_new.mean():.2f}, {lr_new.min():.2g} | {a1:.3f} ± {s1:.3f} | "
              f"{boot(lr_new, [r.get('LR_warp', np.nan) for r in rs])} | "
              f"{np.max(np.abs(lr_new - lr_fit)):.3g} | {same}/{len(rs)} |")
    if Path(direct).exists():
        print("\n### Genuine bootstrap, B = 99, study replications (profile statistic)\n")
        print("| r | LR (fits) | LR | p χ² | p bootstrap | bootstrap q95 | bootstrap mean | B valid |")
        print("|---|---|---|---|---|---|---|---|")
        for r in sorted(_read(direct), key=lambda d: -d["LR"]):
            print(f"| {int(r['r'])} | {r['LR_fits']:.2f} | {r['LR']:.2f} | {r['p_asym']:.4f} | "
                  f"{r['p_boot']:.2f} | {r['boot_q95']:.2f} | {r['boot_mean']:.2f} | "
                  f"{int(r['B_valid'])} |")


def _chi2q(df):
    from scipy.stats import chi2
    return float(chi2.ppf(0.95, df))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--quick", action="store_true", help="4 replications per cell, no files")
    ap.add_argument("--direct", action="store_true", help="the B = 99 check only")
    ap.add_argument("--summarise", action="store_true", help="tables from the CSV files only")
    ap.add_argument("--only", default="", help="comma-separated models (hmc_in, hmc_dn)")
    ap.add_argument("--profile", action="store_true",
                    help="the cells of PROFILE_PLAN → results/lr_study_profile.csv")
    ap.add_argument("--direct-study", action="store_true",
                    help="B = 99 on replications of the study's HMC-DN Markov null, N = 500")
    args = ap.parse_args()
    logging.disable(logging.WARNING)
    if args.summarise:
        summarise()
        summarise_profile()
        return
    if args.direct:
        jobs = [("hmc_in", "state-null", 500, r, 99, "direct") for r in range(200)]
        _run(_job_direct, jobs, args.jobs, OUT / "lr_direct.csv")
        summarise()
        return
    if args.direct_study:
        jobs = [("hmc_dn", "markov-null", 500, r, 99, "study") for r in DIRECT_STUDY_R]
        _run(_job_direct, jobs, args.jobs, OUT / "lr_direct_dn.csv")
        summarise()
        return
    only = set(filter(None, args.only.split(",")))
    plan = PROFILE_PLAN if args.profile else PLAN
    jobs = []
    for (mdl, scen), (Ns, R) in plan.items():
        if only and mdl not in only:
            continue
        for N in Ns:
            for r in range(4 if args.quick else R):
                jobs.append((mdl, scen, N, r, "@" not in scen))
    # the slow cells first, so the pool stays busy
    jobs.sort(key=lambda j: -(j[2] * (8 if j[0] == "hmc_dn" else 1)))
    name = "lr_study_profile" if args.profile else "lr_study"
    path = OUT / (f"{name}_quick.csv" if args.quick else f"{name}.csv")
    _run(_job, jobs, args.jobs, path)
    summarise(path)


if __name__ == "__main__":
    main()
