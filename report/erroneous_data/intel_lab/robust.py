"""
robust.py — step 3 of the Intel Lab study: estimation on a window that mixes
clean and failing readings (erroneous data).

Window of each mote: the 4 days before its first reading > 60 °C and the
day after it (epochs FS − 4·2880 … FS + 2880), raw readings (dequantised,
suspect readings kept), absent epochs as NaN. Model and K: those selected by
fit_clean.py. Every fit is ICE ("available", best iterate, 50 iterations),
from the k-means warm start of the rows left observed (``init = "kmeans"``,
seed 0) unless stated:

``raw``         the window as it is;
``oracle``      suspect rows set to NaN (the threshold truth);
``oracle_ext``  suspect and out-of-range-climb rows set to NaN (the best
                truth available: il_common.truth_labels);
``fm``          ``robust_estimate`` (α = 1e-3, sequential flags, ≤ 10 refits);
``fm_hampel``   the same, ``initial_mask`` = Hampel identifier (t = 4,
                centred window of 11 observed rows);
``fm_thr60``    the same, ``initial_mask`` = the 60 °C threshold (a
                sensor-level flag, re-tested by the model);
``raw_cs``, ``fm_cs``  ``raw`` and ``fm`` started from the clean-window model
                (``init = "model"``) instead of k-means.

Scores: parameters (states sorted by mean), a "failure state" (a state whose
mean exceeds the fit-window maximum by more than 2 °C), the mask against the
labels, and the MPM classification of the normal-operation rows against the
``oracle_ext`` fit and against the clean-window model (Hungarian alignment).

Usage (from the repository root, after fit_clean.py)
----------------------------------------------------
    .venv/bin/python report/erroneous_data/intel_lab/robust.py --jobs 4
"""
from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import il_common as C
from pmcprg.pmc import classify, ice, robust_estimate

WIN_BEFORE_D = 4.0
WIN_AFTER_D = 1.0
METHODS = ("raw", "oracle", "oracle_ext", "fm", "fm_hampel", "fm_thr60", "raw_cs", "fm_cs")
FLAG_CFG = {"alpha": C.MAIN_ALPHA, "sequential": True}
MAX_ROUNDS = 10


def window(d: dict) -> tuple[int, int]:
    fs = d["first_suspect"]
    return int(fs - round(WIN_BEFORE_D * C.EPD)), int(fs + round(WIN_AFTER_D * C.EPD))


def task(spec: dict) -> dict:
    logging.getLogger("pmcprg").setLevel(logging.ERROR)
    m, kind, K, meth = spec["mote"], spec["kind"], spec["K"], spec["method"]
    d = C.load_mote(m, spec["data"])
    lo, hi = window(d)
    lab = C.truth_labels(d, lo, hi)
    Y = d["y"][lo - 1:hi].copy()
    clean_model = C.load_model(C.model_path(m, kind, K))
    template = C.rc.initial_model(kind, K, Y)
    cfg_km = C.ice_cfg(kind, init="kmeans", kmeans_seed=0)
    cfg_model = C.ice_cfg(kind)
    t0 = time.perf_counter()
    out = {"mote": m, "kind": kind, "K": K, "method": meth, "lo": lo, "hi": hi}
    n_fits, converged, mask = 1, np.nan, np.zeros(Y.size, bool)
    if meth in ("raw", "oracle", "oracle_ext", "raw_cs"):
        if meth == "oracle":
            mask = lab["suspect"].copy()
        elif meth == "oracle_ext":
            mask = lab["suspect"] | lab["climb"]
        Ym = Y.copy()
        Ym[mask] = np.nan
        if meth == "raw_cs":
            model, _ = ice(clean_model, Ym, ice_cfg=cfg_model)
        else:
            model, _ = ice(template, Ym, ice_cfg=cfg_km)
    else:
        init_mask = None
        if meth == "fm_hampel":
            init_mask = C.hampel(Y, 4.0)
        elif meth == "fm_thr60":
            raw = d["raw"][lo - 1:hi]
            init_mask = np.isfinite(raw) & (raw > C.T_HI)
        if meth == "fm_cs":
            R = robust_estimate(clean_model, Y, cfg_model, flag_cfg=FLAG_CFG,
                                max_rounds=MAX_ROUNDS)
        else:
            R = robust_estimate(template, Y, cfg_km, flag_cfg=FLAG_CFG,
                                max_rounds=MAX_ROUNDS, initial_mask=init_mask)
        model, mask = R.model, R.mask
        n_fits, converged = R.n_fits, R.converged
        out["initial_mask"] = int(R.masks[0].sum())
        out["masks_per_fit"] = "|".join(str(int(x.sum())) for x in R.masks)
    Ym = Y.copy()
    Ym[mask] = np.nan
    X, _, ll = classify(model, Ym)
    out["seconds"] = time.perf_counter() - t0
    out.update({"n_fits": n_fits, "converged": converged, "loglik": ll,
                "n_masked": int(mask.sum())})
    ob = lab["observed"]
    s, cl = lab["suspect"], lab["climb"]
    out.update({"n_obs": int(ob.sum()), "n_suspect": int(s.sum()), "n_climb": int(cl.sum()),
                "n_ambig": int(lab["ambig"].sum()), "n_normal": int(lab["normal"].sum()),
                "mask_recall_suspect": float((mask & s).sum() / max(s.sum(), 1)),
                "mask_recall_climb": float((mask & cl).sum() / max(cl.sum(), 1)),
                "mask_normal": int((mask & lab["normal"]).sum()),
                "mask_ambig": int((mask & lab["ambig"]).sum())})
    p = C.params(model)
    top = float(np.nanmax(d["raw"][:C.FIT_END])) + 2.0
    fail = np.nonzero(p["mean"] > top)[0]
    out.update({"mean": C.fmt_vec(p["mean"]), "sd": C.fmt_vec(p["sd"]),
                "pi": C.fmt_vec(p["pi"], "{:.3f}"), "stay": C.fmt_vec(p["stay"], "{:.4f}"),
                "copulas": " / ".join(f"{f}({t:.3f})" for f, t in zip(p.get("fam", []), p.get("tau", []))),
                "fail_state": bool(fail.size),
                "fail_mean": float(p["mean"][fail[-1]]) if fail.size else np.nan,
                "fail_sd": float(p["sd"][fail[-1]]) if fail.size else np.nan,
                "fail_pi": float(p["pi"][fail[-1]]) if fail.size else np.nan,
                "fail_stay": float(p["stay"][fail[-1]]) if fail.size else np.nan})
    rank = np.empty(K, int)
    rank[p["order"]] = np.arange(K)
    out["labels"] = rank[np.asarray(X, int)].astype(np.int8)
    out["mask"] = np.packbits(mask)
    out["params"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.items()}
    return out


def run(args):
    sel = C.selected_models()
    specs = [{"mote": m, "kind": k, "K": sel[(m, k)][0], "method": meth, "data": args.data}
             for m in C.MOTES for k in C.KINDS for meth in METHODS]
    specs.sort(key=lambda s: (s["kind"] != "pmc_state", not s["method"].startswith("fm")))
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.jobs) as ex:
        res = list(ex.map(task, specs))
    wall = time.perf_counter() - t0
    # Classification of the normal-operation rows against oracle_ext and the clean model.
    for m in C.MOTES:
        d = C.load_mote(m, args.data)
        lo, hi = window(d)
        lab = C.truth_labels(d, lo, hi)
        Yx = d["y"][lo - 1:hi].copy()
        Yx[lab["suspect"] | lab["climb"]] = np.nan
        for k in C.KINDS:
            K, clean_model = sel[(m, k)]
            Xc, _, _ = classify(clean_model, Yx)
            pc = C.params(clean_model)
            rank = np.empty(K, int)
            rank[pc["order"]] = np.arange(K)
            Xc = rank[np.asarray(Xc, int)]
            ref = next(r for r in res if r["mote"] == m and r["kind"] == k and r["method"] == "oracle_ext")
            no = lab["normal"]
            for r in res:
                if r["mote"] != m or r["kind"] != k:
                    continue
                X = r["labels"].astype(int)
                r["agree_oracle_ext"] = C.rc.agreement(ref["labels"].astype(int), X, K, sel=no)
                r["agree_clean_model"] = C.rc.agreement(Xc, X, K, sel=no)
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("labels", "mask", "params")}
                       for r in res])
    order = {m: i for i, m in enumerate(METHODS)}
    df["o"] = df["method"].map(order)
    df = df.sort_values(["mote", "kind", "o"]).drop(columns="o")
    df.to_csv(C.RESULTS / "robust_fits.csv", index=False)
    C.CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(C.CACHE / "robust_labels.npz",
                        **{f"{r['mote']}|{r['kind']}|{r['method']}|labels": r["labels"] for r in res},
                        **{f"{r['mote']}|{r['kind']}|{r['method']}|mask": r["mask"] for r in res})
    C.save_json(C.RESULTS / "robust_info.json", {
        "pmcprg": C.check_import(), "wall_seconds": wall, "jobs": args.jobs,
        "fit_seconds_sum": float(sum(r["seconds"] for r in res)),
        "windows": {str(m): list(window(C.load_mote(m, args.data))) for m in C.MOTES}})
    figures(res, args)


def figures(res, args):
    plt = C.plot_setup()
    show = ("raw", "oracle_ext", "fm", "fm_thr60")
    cols = [C.C_BLUE, C.C_ORANGE, C.C_AQUA, C.C_VIOLET]
    for kind in C.KINDS:
        fig, axs = plt.subplots(len(C.MOTES), len(show), figsize=(13, 2.5 * len(C.MOTES)),
                                sharey="row")
        for r_, m in enumerate(C.MOTES):
            d = C.load_mote(m, args.data)
            lo, hi = window(d)
            e = d["epoch"][lo - 1:hi]
            raw = d["raw"][lo - 1:hi]
            ob = np.isfinite(raw)
            for c_, meth in enumerate(show):
                r = next(x for x in res if x["mote"] == m and x["kind"] == kind and x["method"] == meth)
                n = e.size
                mask = np.unpackbits(r["mask"])[:n].astype(bool)
                lab = r["labels"].astype(int)
                ax = axs[r_, c_]
                t = (e - d["first_suspect"]) / C.EPH
                for s in range(r["K"]):
                    w = ob & ~mask & (lab == s)
                    ax.plot(t[w], raw[w], ".", ms=1.0, color=cols[s], label=f"state {s}")
                w = ob & mask
                ax.plot(t[w], raw[w], ".", ms=1.0, color="#b9b8b3", label="masked")
                ax.axvline(0, color=C.INK, lw=0.6)
                ax.axvline((d["climb_onset"] - d["first_suspect"]) / C.EPH, color=C.INK2, lw=0.6, ls=":")
                ax.set_title(f"mote {m}, {meth}: {r['n_masked']} masked", fontsize=8)
                if r_ == len(C.MOTES) - 1:
                    ax.set_xlabel("hours from first reading > 60 °C")
                if c_ == 0:
                    ax.set_ylabel("°C")
                if r_ == 0 and c_ == 0:
                    ax.legend(markerscale=6, fontsize=6, loc="upper left")
        fig.suptitle(f"{C.KIND_LABEL[kind]}: MPM states (sorted by mean) of each fit on the mixed window",
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(C.FIGURES / f"robust_{kind}.png")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data", default=C.DEFAULT_DATA)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    print("pmcprg:", C.check_import())
    run(args)


if __name__ == "__main__":
    main()
