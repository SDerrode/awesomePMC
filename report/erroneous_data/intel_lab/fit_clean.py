"""
fit_clean.py — step 1 of the Intel Lab study: models of the clean window and
their calibration (erroneous data).

For each mote (48, 22, 47), model (HMC-IN; PMC with state margins) and
K ∈ {2, 3}: ICE (missing_strategy "available", best iterate) from 3 starts
(k-means + 2 multistart draws of report/real_series) on days 1–7 (epochs
1–20 160, dequantised temperature, gaps as NaN). The best start per cell is
kept (exact log p(y_obs)); K is chosen per (mote, model) by BIC.

Calibration: ``predictive_pit`` over days 1–10 (the filter runs from epoch 1,
so every held-out row is predicted from its whole past) and ``pit_checks``
on the held-out days 8–10 (epochs 20 161–28 800) and on the fit days; the
flag counts of ``flag_outliers`` on the held-out rows. Regimes: MPM
classification of days 1–10 and state occupancy by hour of day.

Quadrature: every PMC pass uses the library default ``gap_nodes`` = 64; the
WARNINGs of pmcprg are counted per task and ``quad_error`` is recorded. The
selected PMC's PIT and sequential flags are rerun at 256 nodes (the
``check_*`` columns of pit_checks.csv).

Usage (from the repository root)
--------------------------------
    .venv/bin/python report/erroneous_data/intel_lab/fit_clean.py --jobs 4

writes results/models/*.toml, results/fits.csv, results/selected_models.json,
results/pit_checks.csv, results/regimes_by_hour.csv and figures/clean_*.png.
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import il_common as C
from pmcprg.pmc import classify, flag_outliers, pit_checks, predictive_pit


def task_fit(spec: dict) -> dict:
    wlog = C.capture_warnings()
    d = C.load_mote(spec["mote"], spec["data"])
    Y = d["y"][:C.FIT_END]
    t0 = time.perf_counter()
    model, trace, tag, t_ice = C.fit_start(spec["kind"], spec["K"], Y, spec["start"])
    ll = C.exact_loglik(model, Y)
    n_obs = int(np.isfinite(Y).sum())
    warned = C.warning_summary(wlog)        # the ICE E-steps and the log-likelihood pass
    # The quadrature diagnostic of the returned model on the fit rows (NaN: HMC-IN).
    q = C.quad_check(model, Y)
    return {**{k: spec[k] for k in ("mote", "kind", "K", "start")}, "tag": tag,
            "n_obs": n_obs, "loglik": ll, "bic": C.bic(model, ll, n_obs),
            "k_params": int(C.rc.n_free_params(model)), "iters": len(trace.log_liks),
            "best_iter": int(getattr(trace, "best_iter", -1)),
            "quad_error": q["quad_error"], "quad_limit": q["quad_limit"], **warned,
            "seconds": time.perf_counter() - t0, "raw": model.raw}


def acf1(z: np.ndarray) -> float:
    z = z[np.isfinite(z)]
    z = z - z.mean()
    return float(np.sum(z[1:] * z[:-1]) / np.sum(z * z))


def task_pit(spec: dict) -> dict:
    wlog = C.capture_warnings()
    d = C.load_mote(spec["mote"], spec["data"])
    model = C.load_model(C.model_path(spec["mote"], spec["kind"], spec["K"]))
    Y = d["y"][:C.CLEAN_END]
    t0 = time.perf_counter()
    P = predictive_pit(model, Y, gap_nodes=C.GAP_NODES)
    t_pit = time.perf_counter() - t0
    after_gap = np.r_[False, ~np.isfinite(Y[:-1])] & np.isfinite(Y)
    after_obs = np.r_[False, np.isfinite(Y[:-1])] & np.isfinite(Y)
    fit_rows = np.zeros(Y.size, bool)
    fit_rows[:C.FIT_END] = True
    rows = []
    for part, exclude in (("held-out", fit_rows), ("fit", ~fit_rows)):
        c = pit_checks(P, exclude=exclude)
        keep = np.isfinite(P.pit) & ~exclude
        r = {"mote": spec["mote"], "kind": spec["kind"], "K": spec["K"], "part": part,
             "n": c.n, "ks_stat": c.ks_stat, "ks_p": c.ks_pvalue, "lb_stat": c.lb_stat,
             "lb_p": c.lb_pvalue, "lb2_stat": c.lb2_stat, "lb2_p": c.lb2_pvalue,
             "z_mean": c.z_mean, "z_sd": c.z_sd, "acf1_z": acf1(P.z[keep]),
             "n_after_gap": int((after_gap & keep).sum()),
             "z_sd_after_gap": float(np.std(P.z[after_gap & keep])),
             "z_sd_after_obs": float(np.std(P.z[after_obs & keep])),
             "nonseq_after_gap_0.001": int((P.pvalue[after_gap & keep] < C.MAIN_ALPHA).sum()),
             "log_lik_days_1_10": P.log_lik, "gap_nodes": C.GAP_NODES,
             "pit_seconds": t_pit}
        for a in C.ALPHAS:
            r[f"nonseq_{a:g}"] = int((P.pvalue[keep] < a).sum())
        rows.append(r)
    F = flag_outliers(model, Y, alpha=C.MAIN_ALPHA, sequential=True, gap_nodes=C.GAP_NODES)
    for r, exclude in zip(rows, (fit_rows, ~fit_rows)):
        r[f"seq_{C.MAIN_ALPHA:g}"] = int((F.flagged & ~exclude).sum())
    X, gamma, _ = classify(model, Y)
    warned = C.warning_summary(wlog)        # classify: the batch pass over days 1–10
    # Quadrature diagnostic of the pass over days 1–10 (the grids of predictive_pit).
    q = C.quad_check(model, Y, C.GAP_NODES)
    extra = {"quad_error_days_1_10": q["quad_error"], "quad_limit_days_1_10": q["quad_limit"],
             "batch_log_lik_days_1_10": q["log_lik"], **warned}
    if spec.get("check"):
        # Sensitivity: the same PIT and sequential flags at GAP_NODES_CHECK nodes.
        G = C.GAP_NODES_CHECK
        P2 = predictive_pit(model, Y, gap_nodes=G)
        F2 = flag_outliers(model, Y, alpha=C.MAIN_ALPHA, sequential=True, gap_nodes=G)
        q2 = C.quad_check(model, Y, G)
        extra.update({"check_gap_nodes": G, "check_log_lik_days_1_10": P2.log_lik,
                      "check_quad_error_days_1_10": q2["quad_error"]})
        for r, exclude in zip(rows, (fit_rows, ~fit_rows)):
            keep = np.isfinite(P2.pit) & ~exclude
            r.update({"check_z_sd_after_gap": float(np.std(P2.z[after_gap & keep])),
                      "check_nonseq_0.001": int((P2.pvalue[keep] < C.MAIN_ALPHA).sum()),
                      f"check_seq_{C.MAIN_ALPHA:g}": int((F2.flagged & ~exclude).sum()),
                      "check_seq_same_rows": bool(np.array_equal(F2.flagged & ~exclude,
                                                                 F.flagged & ~exclude))})
    for r in rows:
        r.update(extra)
    out = {"rows": rows, "pit": P.pit.tolist(), "z": P.z.tolist(), "labels": X.tolist(),
           "order": C.params(model)["order"].tolist()}
    return out


def run(args):
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    specs = [{"mote": m, "kind": k, "K": K, "start": s, "data": args.data}
             for m in C.MOTES for k in C.KINDS for K in C.KS for s in range(C.N_STARTS)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.jobs) as ex:
        fits = list(ex.map(task_fit, specs))
    t_fit = time.perf_counter() - t0
    df = pd.DataFrame([{k: v for k, v in f.items() if k != "raw"} for f in fits])
    best = {}
    for f in fits:
        key = (f["mote"], f["kind"], f["K"])
        if key not in best or f["loglik"] > best[key]["loglik"]:
            best[key] = f
    df["best"] = [best[(r.mote, r.kind, r.K)]["start"] == r.start for r in df.itertuples()]
    for (m, k, K), f in best.items():
        model = C.PMCModel.from_dict(f["raw"])
        model.save(C.model_path(m, k, K))
    sel = {}
    for m in C.MOTES:
        for k in C.KINDS:
            cand = [best[(m, k, K)] for K in C.KS]
            sel[f"{m}|{k}"] = int(min(cand, key=lambda f: f["bic"])["K"])
    df["selected"] = [bool(r.best and sel[f"{r.mote}|{r.kind}"] == r.K) for r in df.itertuples()]
    summ = []
    for r in df.itertuples():
        p = C.params(C.PMCModel.from_dict(best[(r.mote, r.kind, r.K)]["raw"]))
        summ.append({"mean": C.fmt_vec(p["mean"]), "sd": C.fmt_vec(p["sd"]),
                     "pi": C.fmt_vec(p["pi"]), "stay": C.fmt_vec(p["stay"], "{:.4f}"),
                     "copulas": " / ".join(f"{f}({t:.3f})" for f, t in zip(p.get("fam", []), p.get("tau", [])))})
    df = pd.concat([df, pd.DataFrame(summ)], axis=1)
    df.to_csv(C.RESULTS / "fits.csv", index=False)
    C.save_json(C.RESULTS / "selected_models.json", sel)

    # The selected PMC models are also checked at GAP_NODES_CHECK nodes.
    pspecs = [{"mote": m, "kind": k, "K": K, "data": args.data,
               "check": k == "pmc_state" and sel[f"{m}|{k}"] == K}
              for m in C.MOTES for k in C.KINDS for K in C.KS]
    t1 = time.perf_counter()
    with ProcessPoolExecutor(args.jobs) as ex:
        pits = list(ex.map(task_pit, pspecs))
    t_pit = time.perf_counter() - t1
    rows = [r for p in pits for r in p["rows"]]
    pc = pd.DataFrame(rows)
    pc["selected"] = [sel[f"{r.mote}|{r.kind}"] == r.K for r in pc.itertuples()]
    pc.to_csv(C.RESULTS / "pit_checks.csv", index=False)
    C.save_json(C.RESULTS / "fit_clean_info.json", {
        "pmcprg": C.check_import(), "pmcprg_commit": C.git_commit(), "fit_seconds": t_fit, "pit_seconds": t_pit,
        "jobs": args.jobs, "selected": sel})
    figures(pspecs, pits, sel, args)


def figures(pspecs, pits, sel, args):
    plt = C.plot_setup()
    # PIT histograms and normal-score ACF on the held-out days, selected K.
    fig, axs = plt.subplots(len(C.MOTES), 3, figsize=(10, 2.3 * len(C.MOTES)))
    col = {"hmc_in": C.C_BLUE, "pmc_state": C.C_ORANGE}
    for r, m in enumerate(C.MOTES):
        for spec, p in zip(pspecs, pits):
            if spec["mote"] != m or sel[f"{m}|{spec['kind']}"] != spec["K"]:
                continue
            k = spec["kind"]
            pit = np.asarray(p["pit"], float)[C.FIT_END:]
            z = np.asarray(p["z"], float)[C.FIT_END:]
            ci = 0 if k == "hmc_in" else 1
            ax = axs[r, ci]
            ax.hist(pit[np.isfinite(pit)], bins=20, range=(0, 1), density=True,
                    color=col[k], edgecolor="white", linewidth=0.6)
            ax.axhline(1.0, color=C.INK2, lw=0.8, ls="--")
            ax.set_title(f"mote {m}: {C.KIND_LABEL[k]} K={spec['K']}, held-out PIT")
            ax.set_xlim(0, 1)
            zz = z[np.isfinite(z)] - np.nanmean(z)
            lags = np.arange(1, 61)
            acf = [np.sum(zz[h:] * zz[:-h]) / np.sum(zz * zz) for h in lags]
            axs[r, 2].plot(lags, acf, color=col[k], lw=1.2, label=C.KIND_LABEL[k])
        axs[r, 2].axhline(0, color=C.INK2, lw=0.6)
        axs[r, 2].set_title(f"mote {m}: ACF of the normal scores")
        axs[r, 2].set_xlabel("lag (observed rows)")
        axs[r, 2].legend()
    fig.tight_layout()
    fig.savefig(C.FIGURES / "clean_pit.png")
    plt.close(fig)

    # Regimes: 3 days of the held-out window coloured by MPM state + hour occupancy.
    occ_rows = []
    fig, axs = plt.subplots(len(C.MOTES), 2, figsize=(11, 2.4 * len(C.MOTES)),
                            gridspec_kw={"width_ratios": [3, 1]})
    for r, m in enumerate(C.MOTES):
        d = C.load_mote(m, args.data)
        e = d["epoch"][:C.CLEAN_END]
        hour = C.epoch_time(e).hour.to_numpy()
        y = d["y"][:C.CLEAN_END]
        for spec, p in zip(pspecs, pits):
            if spec["mote"] != m or sel[f"{m}|{spec['kind']}"] != spec["K"]:
                continue
            k, K = spec["kind"], spec["K"]
            order = list(p["order"])
            rank = np.empty(K, int)
            rank[order] = np.arange(K)
            lab = rank[np.asarray(p["labels"], int)]
            ob = np.isfinite(y)
            for s in range(K):
                for h in range(24):
                    sel_h = ob & (hour == h)
                    occ_rows.append({"mote": m, "kind": k, "K": K, "state": s, "hour": h,
                                     "share": float(np.mean(lab[sel_h] == s)) if sel_h.any() else np.nan})
            if k == "pmc_state":
                w = slice(C.FIT_END, C.CLEAN_END)
                ax = axs[r, 0]
                cols = [C.C_BLUE, C.C_ORANGE, C.C_AQUA]
                for s in range(K):
                    msk = ob[w] & (lab[w] == s)
                    ax.plot(C.epoch_time(e[w][msk]), y[w][msk], ".", ms=1.2, color=cols[s],
                            label=f"PMC state {s}")
                ax.set_title(f"mote {m}: held-out days 8–10, MPM states of the PMC (K={K})")
                ax.set_ylabel("°C")
                ax.legend(markerscale=6, loc="upper right")
            ls = "-" if k == "pmc_state" else ":"
            ax2 = axs[r, 1]
            cols = [C.C_BLUE, C.C_ORANGE, C.C_AQUA]
            for s in range(K):
                sh = [o["share"] for o in occ_rows if o["mote"] == m and o["kind"] == k and o["state"] == s]
                ax2.plot(range(24), sh, ls, color=cols[s], lw=1.2,
                         label=f"{C.KIND_LABEL[k]} {s}")
            ax2.set_title("state share by hour of day")
            ax2.set_xlabel("hour")
            ax2.set_ylim(0, 1)
            ax2.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "clean_regimes.png")
    plt.close(fig)
    pd.DataFrame(occ_rows).to_csv(C.RESULTS / "regimes_by_hour.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data", default=C.DEFAULT_DATA)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    print("pmcprg:", C.check_import())
    run(args)


if __name__ == "__main__":
    main()
