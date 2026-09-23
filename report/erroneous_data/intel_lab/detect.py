"""
detect.py — step 2 of the Intel Lab study: detection of the failing readings
by the fixed clean-window models, against simple baselines (erroneous data).

The models selected by fit_clean.py (fitted on days 1–7) are held fixed.
``flag_outliers`` runs on the raw readings of epochs 28 801–102 706 (days
10–36: normal operation, battery failure, saturated readings), dequantised,
the suspect readings put back, absent epochs as NaN:

* sequential (innovation gating) and non-sequential;
* per-row α ∈ {1e-2, 1e-3, 1e-4}, and the Benjamini–Hochberg correction at
  FDR α ∈ {1e-2, 1e-3}.

Baselines on the same rows: the temperature half of the suspect rule (60 °C,
the reference that defines the first threshold hit), fixed upper thresholds
35 / 40 / 50 °C, the clean envelope (fit-window range ± 2 °C), a Hampel
identifier (centred window of 11 observed rows, t = 3 / 4 / 5) and a rolling
robust z-score (trailing 24 h, t = 3 / 4 / 5).

Scores against the labels of il_common.truth_labels (suspect; out-of-range
climb before the first suspect; the ambiguous 24 h before it; normal
operation): recall and precision against ``suspect``, flags per label, false
alarms in normal operation, lead times.

Usage (from the repository root, after fit_clean.py)
----------------------------------------------------
    .venv/bin/python report/erroneous_data/intel_lab/detect.py --jobs 4
"""
from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import il_common as C
from pmcprg.pmc import flag_outliers

LO = C.CLEAN_END + 1          # first epoch of the detection window


def settings() -> list[tuple[bool, float, str | None]]:
    out = []
    for seq in (False, True):
        out += [(seq, a, None) for a in C.ALPHAS]
        out += [(seq, a, "bh") for a in C.BH_ALPHAS]
    return out


def setting_name(kind: str, seq: bool, alpha: float, corr) -> str:
    return f"{C.KIND_LABEL[kind]} {'seq' if seq else 'nonseq'} {'BH ' if corr else ''}α={alpha:g}"


def task_flag(spec: dict) -> dict:
    logging.getLogger("pmcprg").setLevel(logging.ERROR)
    d = C.load_mote(spec["mote"], spec["data"])
    model = C.load_model(C.model_path(spec["mote"], spec["kind"], spec["K"]))
    Y = d["y"][LO - 1:]
    t0 = time.perf_counter()
    F = flag_outliers(model, Y, alpha=spec["alpha"], sequential=spec["seq"],
                      correction=spec["corr"])
    return {**{k: spec[k] for k in ("mote", "kind", "K", "seq", "alpha", "corr")},
            "flagged": np.packbits(F.flagged), "pvalue": F.pvalue.astype(np.float32),
            "threshold": F.threshold, "n_tests": F.n_tests, "method": F.method,
            "seconds": time.perf_counter() - t0}


def baselines(d: dict) -> dict:
    """{name: (family, param, flag vector)} on the detection window."""
    y = d["y"][LO - 1:]
    raw = d["raw"][LO - 1:]
    e = d["epoch"][LO - 1:]
    ob = np.isfinite(raw)
    out = {"threshold 60 °C (suspect rule, T)": ("threshold", 60.0, ob & ((raw > C.T_HI) | (raw < C.T_LO)))}
    for t in C.FIXED_TS:
        out[f"threshold {t:g} °C"] = ("threshold", t, ob & (raw > t))
    fit = d["raw"][:C.FIT_END]
    lo, hi = np.nanmin(fit) - C.ENVELOPE_MARGIN, np.nanmax(fit) + C.ENVELOPE_MARGIN
    out[f"clean envelope ±{C.ENVELOPE_MARGIN:g} °C"] = ("envelope", C.ENVELOPE_MARGIN,
                                                        ob & ((raw < lo) | (raw > hi)))
    for t in C.HAMPEL_TS:
        out[f"Hampel t={t:g}"] = ("hampel", t, C.hampel(y, t))
    z = C.robust_z(y, e)
    for t in C.ROBZ_TS:
        out[f"robust z 24 h t={t:g}"] = ("robust_z", t, np.nan_to_num(z, nan=0.0) > t)
    return out


def run(args):
    sel = C.selected_models()
    C.CACHE.mkdir(parents=True, exist_ok=True)
    specs = [{"mote": m, "kind": k, "K": sel[(m, k)][0], "seq": s, "alpha": a, "corr": c,
              "data": args.data}
             for m in C.MOTES for k in C.KINDS for (s, a, c) in settings()]
    # Longest first (PMC sequential BH) for a better load balance.
    specs.sort(key=lambda s: (s["kind"] != "pmc_state", not s["seq"], s["corr"] is None))
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.jobs) as ex:
        res = list(ex.map(task_flag, specs))
    t_flag = time.perf_counter() - t0
    rows, store = [], {}
    for m in C.MOTES:
        d = C.load_mote(m, args.data)
        E = d["epoch"].size
        lab = C.truth_labels(d, LO, E)
        fs = d["first_suspect"]
        n = E - LO + 1
        for r in res:
            if r["mote"] != m:
                continue
            flag = np.unpackbits(r["flagged"])[:n].astype(bool)
            name = setting_name(r["kind"], r["seq"], r["alpha"], r["corr"])
            store[f"{m}|{name}"] = flag
            store[f"{m}|{name}|p"] = r["pvalue"]
            rows.append({"mote": m, "method": name, "family": C.KIND_LABEL[r["kind"]],
                         "K": r["K"], "sequential": r["seq"], "alpha": r["alpha"],
                         "correction": r["corr"] or "none", "threshold": r["threshold"],
                         "seconds": r["seconds"], **C.score_flags(flag, lab, fs)})
        t1 = time.perf_counter()
        base = baselines(d)
        t_base = time.perf_counter() - t1
        for name, (fam, par, flag) in base.items():
            store[f"{m}|{name}"] = flag
            rows.append({"mote": m, "method": name, "family": fam, "K": np.nan,
                         "sequential": np.nan, "alpha": par, "correction": "",
                         "threshold": par, "seconds": t_base / len(base),
                         **C.score_flags(flag, lab, fs)})
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS / "detect_scores.csv", index=False)
    np.savez_compressed(C.CACHE / "detect_flags.npz", **{k: v for k, v in store.items()})
    C.save_json(C.RESULTS / "detect_info.json", {
        "pmcprg": C.check_import(), "flag_seconds_wall": t_flag, "jobs": args.jobs,
        "flag_seconds_sum": float(sum(r["seconds"] for r in res)),
        "window_epochs": [LO, int(C.load_mote(C.MOTES[0], args.data)["epoch"].size)],
        "labels": {str(m): {"first_suspect": C.load_mote(m, args.data)["first_suspect"],
                            "climb_onset": C.load_mote(m, args.data)["climb_onset"]}
                   for m in C.MOTES}})
    episodes_table(store, args)
    figures(store, args)


MAIN = [("PMC seq α=0.001", C.C_ORANGE), ("PMC nonseq α=0.001", C.C_YELLOW),
        ("HMC-IN seq α=0.001", C.C_BLUE), ("Hampel t=4", C.C_AQUA),
        ("robust z 24 h t=4", C.C_VIOLET), ("threshold 40 °C", C.C_MAGENTA)]


def episodes_table(store: dict, args):
    """Every alarm episode of the main settings outside the suspect rows, with context."""
    rows = []
    for m in C.MOTES:
        d = C.load_mote(m, args.data)
        E = d["epoch"].size
        lab = C.truth_labels(d, LO, E)
        e = lab["epoch"]
        raw = d["raw"][LO - 1:]
        for name, _ in MAIN:
            flag = store[f"{m}|{name}"] & lab["observed"] & ~lab["suspect"]
            for a, b, nf in C.episodes(e[flag]):
                w = (e >= a) & (e <= b) & lab["observed"]
                before = (e >= a - C.EPH) & (e < a) & lab["observed"]
                zone = ("climb" if lab["climb"][w].any() else "ambiguous" if lab["ambig"][w].any()
                        else "post" if lab["post"][w].any() else "normal")
                ya = raw[w]
                jump = np.nan
                idx = np.nonzero(lab["observed"])[0]
                pos = np.searchsorted(idx, np.nonzero(e == a)[0][0])
                if pos > 0:
                    jump = raw[idx[pos]] - raw[idx[pos - 1]]
                rows.append({"mote": m, "method": name, "zone": zone,
                             "start": str(C.epoch_time([a])[0])[:16], "start_epoch": a,
                             "hours": (b - a) / C.EPH, "n_flags": nf,
                             "n_obs": int(w.sum()),
                             "y_min": float(np.nanmin(ya)), "y_max": float(np.nanmax(ya)),
                             "median_prev_hour": float(np.nanmedian(raw[before])) if before.any() else np.nan,
                             "jump_at_start": float(jump),
                             "gap_before_h": float((e[idx[pos]] - e[idx[pos - 1]]) / C.EPH) if pos > 0 else np.nan})
    pd.DataFrame(rows).to_csv(C.RESULTS / "alarm_episodes.csv", index=False)


def _raster(ax, e, flags, names_cols):
    for i, (name, col) in enumerate(names_cols):
        f = flags[name]
        ax.plot(C.epoch_time(e[f]), np.full(f.sum(), -i), "|", ms=6, mew=0.8, color=col)
    ax.set_yticks(-np.arange(len(names_cols)))
    ax.set_yticklabels([n for n, _ in names_cols], fontsize=7)
    ax.set_ylim(-len(names_cols) + 0.5, 0.5)
    ax.grid(False)


def figures(store: dict, args):
    plt = C.plot_setup()
    for zoom in (False, True):
        fig, axs = plt.subplots(2 * len(C.MOTES), 1, figsize=(11, 3.3 * len(C.MOTES)),
                                gridspec_kw={"height_ratios": [2.2, 1.3] * len(C.MOTES)})
        for r, m in enumerate(C.MOTES):
            d = C.load_mote(m, args.data)
            E = d["epoch"].size
            lab = C.truth_labels(d, LO, E)
            e = lab["epoch"]
            raw = d["raw"][LO - 1:]
            fs, co = d["first_suspect"], d["climb_onset"]
            if zoom:
                w = (e >= fs - 36 * C.EPH) & (e <= fs + 6 * C.EPH)
            else:
                w = np.ones(e.size, bool)
            ax = axs[2 * r]
            ok = w & lab["observed"] & ~lab["suspect"]
            su = w & lab["suspect"]
            ax.plot(C.epoch_time(e[ok]), raw[ok], ".", ms=1.0, color=C.C_BLUE, label="raw, not suspect")
            ax.plot(C.epoch_time(e[su]), raw[su], ".", ms=1.0, color=C.C_RED, label="suspect (> 60 °C)")
            ax.axvspan(C.epoch_time([co])[0], C.epoch_time([fs])[0], color=C.C_YELLOW, alpha=0.18, lw=0,
                       label="out-of-range climb")
            if zoom:
                ax.axvspan(C.epoch_time([fs - C.AMBIG_H * C.EPH])[0], C.epoch_time([co])[0],
                           color=GREY, alpha=0.25, lw=0, label="ambiguous 24 h")
            ax.axvline(C.epoch_time([fs])[0], color=C.INK, lw=0.8)
            ax.set_ylabel("°C")
            ax.set_title(f"mote {m}: raw temperature, epochs {int(e[w][0])}–{int(e[w][-1])}"
                         f" (first reading > 60 °C: {str(C.epoch_time([fs])[0])[:16]})")
            if zoom:
                ax.set_ylim(10, 130)
            ax.legend(markerscale=6, loc="upper left", fontsize=7)
            flags = {name: store[f"{m}|{name}"] & w for name, _ in MAIN}
            _raster(axs[2 * r + 1], e, flags, MAIN)
            axs[2 * r + 1].set_xlim(ax.get_xlim())
        fig.tight_layout()
        fig.savefig(C.FIGURES / ("detect_zoom.png" if zoom else "detect_overview.png"))
        plt.close(fig)

    # p-values of the PMC and HMC-IN (non-sequential) around the failure.
    fig, axs = plt.subplots(len(C.MOTES), 1, figsize=(11, 2.4 * len(C.MOTES)))
    for r, m in enumerate(C.MOTES):
        d = C.load_mote(m, args.data)
        e = d["epoch"][LO - 1:]
        fs = d["first_suspect"]
        w = (e >= fs - 36 * C.EPH) & (e <= fs + 6 * C.EPH)
        ax = axs[r]
        for name, col in (("HMC-IN nonseq α=0.001", C.C_BLUE), ("PMC nonseq α=0.001", C.C_ORANGE)):
            p = store[f"{m}|{name}|p"].astype(float)
            ok = w & np.isfinite(p)
            ax.plot(C.epoch_time(e[ok]), -np.log10(np.maximum(p[ok], 1e-300)), ".", ms=1.2,
                    color=col, label=name.replace(" α=0.001", ""))
        ax.axhline(3, color=C.INK2, lw=0.8, ls="--")
        ax.axvline(C.epoch_time([fs])[0], color=C.INK, lw=0.8)
        ax.axvline(C.epoch_time([d["climb_onset"]])[0], color=C.INK2, lw=0.8, ls=":")
        ax.set_yscale("symlog", linthresh=10)
        ax.set_ylim(0, 400)
        ax.set_ylabel("−log10 p")
        ax.set_title(f"mote {m}: predictive p-values around the failure (dashed: α = 1e-3;"
                     " dotted: climb onset; solid: first reading > 60 °C)")
        ax.legend(markerscale=6, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "detect_pvalues.png")
    plt.close(fig)
    false_alarm_figure(store, args, plt)


def false_alarm_figure(store: dict, args, plt):
    """The largest normal-operation episodes of the PMC (sequential, α = 1e-3), ±3 h."""
    ep = pd.read_csv(C.RESULTS / "alarm_episodes.csv")
    ep = ep[(ep.zone == "normal") & (ep.method == "PMC seq α=0.001")]
    ep = ep.sort_values("n_flags", ascending=False).head(6).sort_values(["mote", "start_epoch"])
    if ep.empty:
        return
    n = len(ep)
    fig, axs = plt.subplots((n + 1) // 2, 2, figsize=(11, 2.4 * ((n + 1) // 2)), squeeze=False)
    for ax, r in zip(axs.ravel(), ep.itertuples()):
        d = C.load_mote(r.mote, args.data)
        e = d["epoch"][LO - 1:]
        raw = d["raw"][LO - 1:]
        a = r.start_epoch - 3 * C.EPH
        b = r.start_epoch + (r.hours + 3) * C.EPH
        w = (e >= a) & (e <= b) & np.isfinite(raw)
        t = (e[w] - r.start_epoch) / C.EPH
        ax.plot(t, raw[w], ".", ms=1.2, color=C.C_BLUE, label="raw")
        for name, col, dy in (("PMC seq α=0.001", C.C_ORANGE, 0.0),
                              ("PMC nonseq α=0.001", C.C_YELLOW, 0.4),
                              ("HMC-IN seq α=0.001", C.C_VIOLET, 0.8)):
            f = store[f"{r.mote}|{name}"][w]
            lo_y = np.nanmin(raw[w]) - 0.6 - dy
            ax.plot(t[f], np.full(f.sum(), lo_y), "|", ms=5, color=col,
                    label=name.replace(" α=0.001", ""))
        ax.set_title(f"mote {r.mote}, {r.start}: {r.n_flags} sequential PMC flags", fontsize=8)
        ax.set_xlabel("hours from the first flag")
        ax.set_ylabel("°C")
    axs.ravel()[0].legend(markerscale=4, fontsize=6, loc="upper left")
    for ax in axs.ravel()[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "detect_false_alarms.png")
    plt.close(fig)


GREY = "#8a8985"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data", default=C.DEFAULT_DATA)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--figures-only", action="store_true",
                    help="episodes table and figures from results/cache/detect_flags.npz")
    args = ap.parse_args()
    print("pmcprg:", C.check_import())
    if args.figures_only:
        with np.load(C.CACHE / "detect_flags.npz") as z:
            store = {k: z[k] for k in z.files}
        episodes_table(store, args)
        figures(store, args)
    else:
        run(args)


if __name__ == "__main__":
    main()
