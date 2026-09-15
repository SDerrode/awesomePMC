"""
summarise.py — tables and figures of the A4 real-series study, from the result
CSVs only (no estimation).

Prints every table quoted in ``README.md`` (Markdown) and writes them to
``<results>/tables.md``; figures go to ``--figures``. The two time-series
figures read their data extracts from ``--extracts`` (outside the repository:
they contain observations).

Usage (repository root)::

    .venv/bin/python report/real_series/summarise.py
    .venv/bin/python report/real_series/summarise.py --results report/out/real_series/quick/results \
        --figures report/out/real_series/quick/figures --extracts report/out/real_series/quick/figure_data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent

# Categorical slots (fixed order) and text inks.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dddcd6"
METHOD_COLOR = {"pmc": C[0], "hmc": C[6], "ar1": C[1], "linear": C[2], "locf": C[3]}
STRAT_COLOR = {"available": C[0], "impute": C[1], "sem": C[2], "complete": C[6]}

OUT: list[str] = []


def emit(text: str = "") -> None:
    print(text)
    OUT.append(text)


def md(df: pd.DataFrame, fmt: dict | None = None, default: str = "{:.3f}") -> str:
    """Markdown table; ``fmt`` maps a column to a format string or a callable."""
    fmt = fmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            f = fmt.get(c, default)
            if isinstance(v, (float, np.floating)):
                if not np.isfinite(v):
                    cells.append("–")
                    continue
                if not callable(f) and ":d" in f:
                    v = int(round(v))
                cells.append(f(v) if callable(f) else f.format(v))
            elif isinstance(v, (int, np.integer)) and c in fmt:
                cells.append(fmt[c](v) if callable(fmt[c]) else fmt[c].format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def section(title: str) -> None:
    emit("")
    emit(f"## {title}")
    emit("")


def style(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def best_start(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Row with the highest log-likelihood in every group of ``keys``."""
    df = df[df.status == "ok"]
    return df.loc[df.groupby(keys).ll.idxmax()].reset_index(drop=True)


def label(r) -> str:
    return f"{r['model']} K={int(r['K'])}"


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def overview(fits: pd.DataFrame, miss: pd.DataFrame | None, info: dict) -> None:
    section("Series")
    f = fits.drop_duplicates("series")[["series", "N", "n_missing"]].copy()
    f["missing_pct"] = 100 * f.n_missing / f.N
    emit(md(f, {"N": "{:d}", "n_missing": "{:d}", "missing_pct": "{:.2f}"}))
    emit("")
    walls = info.get("this_invocation_wall_s", {})
    emit(f"Campaign: {info.get('date', '?')}, {info.get('jobs')} processes; sum of task wall "
         f"times {info.get('sum_task_wall_s', float('nan')) / 60:.1f} min; last invocation "
         f"{walls.get('total_s', float('nan')) / 60:.1f} min wall (tasks already cached are not re-run).")


def failures(fits: pd.DataFrame, errors: pd.DataFrame | None) -> None:
    section("Failures, warnings and non-monotone traces")
    bad = fits[fits.status != "ok"]
    emit(f"Fits: {len(fits)}; failed: {len(bad)}; non-finite parameters: "
         f"{int((fits.finite_params == False).sum())}.")  # noqa: E712
    if len(bad):
        emit(md(bad[["series", "K", "model", "strategy", "start", "error"]]))
    if errors is not None and len(errors):
        emit(f"Crashed tasks: {len(errors)}")
        emit(md(errors[["task", "error"]]))
    ok = fits[fits.status == "ok"]
    g = ok.groupby("strategy").agg(
        fits=("ll", "size"),
        with_ll_decrease=("n_ll_decreases", lambda s: int((s > 0).sum())),
        median_decreases=("n_ll_decreases", "median"),
        max_single_decrease=("max_ll_decrease", "max"),
        with_warnings=("n_warnings", lambda s: int((s > 0).sum()))).reset_index()
    emit("")
    emit("ICE/SEM traces that decrease at least once (SEM is not monotone by design):")
    emit("")
    emit(md(g, {"fits": "{:d}", "with_ll_decrease": "{:d}", "with_warnings": "{:d}",
                "median_decreases": "{:.0f}", "max_single_decrease": "{:.1f}"}))
    if "ll_trace_max" in ok.columns:
        ice_fits = ok[ok.strategy != "sem"]
        below = ice_fits.ll_trace_max - ice_fits.ll_trace_last
        emit("")
        emit(f"ICE fits whose returned model is more than 1 nat below the best iterate of their own "
             f"trace (early stop after `patience` regressions, or `max_iter`): "
             f"{int((below > 1).sum())} of {len(ice_fits)}; median gap among them "
             f"{below[below > 1].median():.1f} nats, largest {below.max():.1f}.")
    import re
    kinds = {}
    for w in ok.warnings.dropna():
        seen = set()
        for k in str(w).split(" || "):
            key = "ICE iter N: log-lik regressed" if k.startswith("ICE iter") else re.sub(r"-?\d+\.\d+", "x", k)
            key = key[:100]
            if key not in seen:
                kinds[key] = kinds.get(key, 0) + 1
                seen.add(key)
    if kinds:
        emit("")
        emit("Distinct library warnings (number of fits with at least one; numbers replaced by x; "
             "only the first three distinct messages of a fit are kept):")
        emit("")
        for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]):
            emit(f"* {v} × `{k}`")


def bic_selection(fits: pd.DataFrame) -> None:
    section("Model and K selection (ICE \"available\" or complete data, best start)")
    emit("Best start = highest log p(y_obs) among the starts of the cell; `degenerate`: a state with "
         "stationary probability below 0.5 %, or a margin "
         "standard deviation below 1 % of the standard deviation of the series; `nd` rows: best "
         "non-degenerate start when the best start is degenerate. dBIC relative to the lowest "
         "non-degenerate BIC of the series.")
    emit("")
    f = fits[fits.strategy.isin(["available", "complete"]) & (fits.status == "ok")]
    b = best_start(f, ["series", "K", "model"])
    nd = best_start(f[~f.degenerate], ["series", "K", "model"])
    extra = []
    for _, r in b[b.degenerate].iterrows():
        q = nd[(nd.series == r.series) & (nd.K == r.K) & (nd.model == r.model)]
        if len(q):
            extra.append(q.iloc[0].to_dict() | {"model": q.iloc[0].model + " (nd)"})
    b = pd.concat([b, pd.DataFrame(extra)], ignore_index=True) if extra else b
    ref = b[~b.degenerate.astype(bool)].groupby("series").bic.min()
    b["dBIC"] = b.bic - b.series.map(ref)
    b = b.sort_values(["series", "bic"])
    for series, g in b.groupby("series", sort=False):
        emit(f"**{series}** (observed n = {int(g.N.iloc[0] - g.n_missing.iloc[0])})")
        emit("")
        cols = ["model", "K", "start", "ll", "n_params", "bic", "dBIC", "degenerate", "min_margin_sd",
                "mean_sorted", "sd_sorted", "copulas", "margins"]
        emit(md(g[cols], {"ll": "{:.1f}", "bic": "{:.1f}", "dBIC": "{:.1f}", "n_params": "{:d}",
                          "K": "{:d}", "start": "{:d}", "min_margin_sd": "{:.2g}"}))
        emit("")
    deg = fits[(fits.status == "ok")].groupby("series").degenerate.agg(["sum", "size"]).reset_index()
    emit("Degenerate fits per series (all strategies and starts):")
    emit("")
    emit(md(deg.rename(columns={"sum": "degenerate", "size": "fits"}), {"degenerate": "{:d}", "fits": "{:d}"}))
    emit("")


def strategies(fits: pd.DataFrame) -> pd.DataFrame:
    section("ICE \"available\" vs ICE \"impute\" vs SEM")
    emit("ΔLL: best log p(y_obs) of the strategy minus the best of \"available\" (same cell); "
         "`k-means ΔLL`: the same from the common k-means start only; spread: max − min (3 starts only) "
         "log-likelihood over the starts; agree: lowest label agreement of a start with the best "
         "fit of the cell (all strategies); time: median estimation time per start.")
    emit("")
    series = ["tsnh4", "huairou", "pamap2_102_gapped"]
    ok = fits[(fits.status == "ok") & fits.series.isin(series)]
    rows = []
    for (s, K, m), g in ok.groupby(["series", "K", "model"]):
        av = g[g.strategy == "available"]
        if av.empty:
            continue
        best_av = av.ll.max()
        km_av = av[av.start == 0].ll.max()
        for strat, h in g.groupby("strategy"):
            rows.append({"series": s, "K": K, "model": m, "strategy": strat, "starts": len(h),
                         "best_ll": h.ll.max(), "dLL": h.ll.max() - best_av,
                         "kmeans_dLL": h[h.start == 0].ll.max() - km_av,
                         "spread": h.ll.max() - h.ll.min() if len(h) >= 3 else np.nan,
                         "agree_min": h.agree_with_best.min(),
                         "degenerate": int(h.degenerate.sum()),
                         "iters": h.n_iter.median(), "time_s": h.runtime_s.median(),
                         "families_best": h.loc[h.ll.idxmax(), "copulas"] if isinstance(
                             h.loc[h.ll.idxmax(), "copulas"], str) else "",
                         "margins_best": h.loc[h.ll.idxmax(), "margins"]})
    t = pd.DataFrame(rows)
    order = {"available": 0, "impute": 1, "sem": 2}
    t["o"] = t.strategy.map(order)
    t = t.sort_values(["series", "K", "model", "o"]).drop(columns="o")
    for s, g in t.groupby("series", sort=False):
        emit(f"**{s}**")
        emit("")
        emit(md(g.drop(columns=["series", "families_best", "margins_best"]),
                {"K": "{:d}", "starts": "{:d}", "best_ll": "{:.1f}", "dLL": "{:+.1f}",
                 "kmeans_dLL": "{:+.1f}", "spread": "{:.1f}", "agree_min": "{:.3f}",
                 "iters": "{:.0f}", "time_s": "{:.1f}", "degenerate": "{:d}"}))
        emit("")
    # the three strategies side by side, pooled
    tav = t[t.strategy == "available"].set_index(["series", "K", "model"]).time_s
    t["time_ratio"] = [r.time_s / tav.loc[(r.series, r.K, r.model)] for r in t.itertuples()]
    cop = t.model != "hmc_in"
    rows = []
    for (strat, is_cop), h in t.groupby([t.strategy, cop]):
        rows.append({"strategy": strat, "models": "copula models" if is_cop else "HMC-IN", "cells": len(h),
                     "median_dLL": h.dLL.median(), "min_dLL": h.dLL.min(), "max_dLL": h.dLL.max(),
                     "median_kmeans_dLL": h.kmeans_dLL.median(),
                     "kmeans_within_1nat": float((h.kmeans_dLL.abs() <= 1).mean()),
                     "cells_3_starts": int((h.starts >= 3).sum()),
                     "median_spread": h.spread.median(),
                     "median_agree_min": h[h.starts >= 3].agree_min.median(),
                     "median_time_ratio": h.time_ratio.median()})
    pooled = pd.DataFrame(rows)
    emit("Pooled over the cells above (time ratio: median time per start / that of \"available\" "
         "in the same cell):")
    emit("")
    emit(md(pooled, {"cells": "{:d}", "median_dLL": "{:+.1f}", "min_dLL": "{:+.1f}",
                     "max_dLL": "{:+.1f}", "median_kmeans_dLL": "{:+.1f}", "kmeans_within_1nat": "{:.2f}",
                     "cells_3_starts": "{:d}",
                     "median_spread": "{:.1f}", "median_agree_min": "{:.3f}",
                     "median_time_ratio": "{:.1f}"}))
    emit("")
    emit("Copula families (and GICE margins) of the best start per strategy, K = 2:")
    emit("")
    fam = t[(t.K == 2) & (t.model != "hmc_in")][["series", "model", "strategy", "families_best",
                                                   "margins_best"]]
    emit(md(fam))
    return t


def runtime_table(fits: pd.DataFrame, tasks: pd.DataFrame | None) -> None:
    section("Runtime")
    ok = fits[fits.status == "ok"]
    g = ok.groupby(["series", "strategy"]).agg(fits=("runtime_s", "size"),
                                              total_min=("runtime_s", lambda s: s.sum() / 60),
                                              median_s=("runtime_s", "median"),
                                              max_s=("runtime_s", "max")).reset_index()
    emit(md(g, {"fits": "{:d}", "total_min": "{:.1f}", "median_s": "{:.1f}", "max_s": "{:.1f}"}))
    if tasks is not None:
        emit("")
        tt = tasks.groupby("task").agg(n=("wall_s", "size"), total_min=("wall_s", lambda s: s.sum() / 60)).reset_index()
        emit(md(tt, {"n": "{:d}", "total_min": "{:.1f}"}))
        emit(f"\nSum over tasks: {tasks.wall_s.sum() / 60:.1f} min (single-process equivalent).")


def tsnh4_imputation(fits: pd.DataFrame, imp: pd.DataFrame, figdir: Path) -> None:
    section("tsNH4: imputation at the real gaps (log scale, truth = tsNH4Complete)")
    f = fits[(fits.series == "tsnh4") & (fits.status == "ok")]
    best = best_start(f, ["K", "model", "strategy"])
    keep = []
    for _, r in best.iterrows():
        sel = imp[(imp.model == r.model) & (imp.K == r.K) & (imp.strategy == r.strategy)
                  & (imp.start == r.start) & (imp.method == "pmc_posterior")]
        deg = " [degenerate]" if bool(r.degenerate) else ""
        keep.append(sel.assign(label=f"{r.model} K={r.K} {r.strategy}{deg}", bic=r.bic, deg=bool(r.degenerate)))
    base = imp[imp.strategy == "baseline"].assign(label=lambda d: d.method, bic=np.nan, deg=False)
    allr = pd.concat(keep + [base], ignore_index=True)
    n = allr[allr.label == allr.label.iloc[0]][["gap_class", "n"]]
    emit("Positions per gap class: " + ", ".join(f"{a}: {int(b)}" for a, b in n.values))
    emit("")
    for metric, fmtx in (("rmse", "{:.3f}"), ("crps", "{:.3f}"), ("cov90", "{:.2f}"), ("width90", "{:.2f}")):
        p = allr.pivot_table(index="label", columns="gap_class", values=metric, aggfunc="first")
        p = p[[c for c in ("all", "1-2", "3-20", ">20") if c in p.columns]]
        p = p.sort_values("all").reset_index()
        emit(f"**{metric}** (CRPS of linear/LOCF = MAE, point imputations)" if metric == "crps" else f"**{metric}**")
        emit("")
        emit(md(p, {c: fmtx for c in p.columns if c != "label"}))
        emit("")
    # figure: metric by gap class for a readable subset
    av = allr[(allr.strategy.isin(["available", "baseline"]))]
    lab_best = av[(av.gap_class == "all") & av.bic.notna() & (av.deg == False)].sort_values("bic").label.iloc[0]  # noqa: E712
    hmc = av[(av.gap_class == "all") & av.label.str.startswith("hmc_in")].sort_values("bic").label.iloc[0]
    sel = [lab_best, hmc, "ar1", "linear", "locf"]
    colors = [METHOD_COLOR["pmc"], METHOD_COLOR["hmc"], METHOD_COLOR["ar1"],
              METHOD_COLOR["linear"], METHOD_COLOR["locf"]]
    classes = [c for c in ("1-2", "3-20", ">20") if c in set(av.gap_class)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, metric in zip(axes, ("rmse", "crps")):
        w = 0.8 / len(sel)
        for k, (lb, col) in enumerate(zip(sel, colors)):
            vals = [av[(av.label == lb) & (av.gap_class == c)][metric].iloc[0] for c in classes]
            ax.bar(np.arange(len(classes)) + (k - (len(sel) - 1) / 2) * w, vals, w * 0.9,
                   color=col, label=lb)
        ax.set_xticks(np.arange(len(classes)), [f"gap {c}" for c in classes])
        ax.set_title(metric.upper() + (" (point methods: MAE)" if metric == "crps" else ""),
                     fontsize=9, color=INK)
        style(ax)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("tsNH4, real gaps: error by gap length (log scale)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(figdir / "tsnh4_imputation_by_gap.png", dpi=130)
    plt.close(fig)


def tsnh4_zoom_fig(z: pd.DataFrame, figdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.2))
    t = z.t.to_numpy()
    ax.plot(t, z.y_obs, color=INK, lw=1.0, label="observed")
    m = z.pmc_mean.notna().to_numpy()
    ax.plot(t, np.where(m, z.y_true, np.nan), color=INK2, lw=1.0, ls=":", label="truth in the gaps")
    ax.fill_between(t, z.pmc_q05, z.pmc_q95, where=m, color=C[0], alpha=0.18, lw=0)
    ax.plot(t, z.pmc_mean, color=C[0], lw=1.6, label=f"{z.pmc_model.iloc[0]}: posterior mean, 90 %")
    ax.plot(t, z.ar1_mean, color=C[1], lw=1.4, label="AR(1) smoother")
    ax.plot(t, z.ar1_lo, color=C[1], lw=0.7, ls="--")
    ax.plot(t, z.ar1_hi, color=C[1], lw=0.7, ls="--")
    ax.plot(t, z.linear, color=C[2], lw=1.2, label="linear")
    ax.set_xlabel("time index (10 min)", fontsize=8, color=INK2)
    ax.set_ylabel("log NH4", fontsize=8, color=INK2)
    style(ax)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper left")
    ax.set_title("tsNH4: longest real gap", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(figdir / "tsnh4_longest_gap.png", dpi=130)
    plt.close(fig)


def beijing_cv(cv: pd.DataFrame, fits: pd.DataFrame, figdir: Path) -> None:
    section("Beijing: masked cross-validation (hidden observed values, 5 seeds)")
    emit("Mean (sd) over seeds, scored on the artificially hidden positions against ln(PM2.5) "
         "as reported; the models are refitted (ICE \"available\", k-means start) on each masked "
         "series. agree_obs / agree_hidden: label agreement with the fit on the unmasked series.")
    emit("")
    cv = cv.copy()
    cv["label"] = np.where(cv.K > 0, cv.method + " K=" + cv.K.astype(int).astype(str), cv.method)
    ok = cv[cv.status.fillna("ok") == "ok"]
    g = ok.groupby(["station", "rate", "block", "label"]).agg(
        seeds=("rmse", "size"), rmse=("rmse", "mean"), rmse_sd=("rmse", "std"),
        crps=("crps", "mean"), crps_sd=("crps", "std"), cov90=("cov90", "mean"),
        agree_obs=("agree_ref_observed", "mean"), agree_hidden=("agree_ref_hidden", "mean"),
        fit_s=("fit_s", "median")).reset_index()
    for st, h in g.groupby("station"):
        emit(f"**{st}**")
        emit("")
        emit(md(h.drop(columns="station"), {"rate": "{:.2f}", "block": "{:d}", "seeds": "{:d}",
                                            "rmse": "{:.3f}", "rmse_sd": "{:.3f}", "crps": "{:.3f}",
                                            "crps_sd": "{:.3f}", "cov90": "{:.2f}",
                                            "agree_obs": "{:.3f}", "agree_hidden": "{:.3f}",
                                            "fit_s": "{:.1f}"}))
        emit("")
    failed = cv[cv.status == "failed"]
    if len(failed):
        emit(f"Failed CV fits: {len(failed)}")
        emit(md(failed[["station", "rate", "block", "seed", "method", "error"]]))
    # wins: paired per seed, best copula model vs AR(1)
    emit("Paired comparison per (station, rate, block, seed): share of cells where the PMC/HMC "
         "model has a lower CRPS than the AR(1) smoother:")
    emit("")
    piv = ok.pivot_table(index=["station", "rate", "block", "seed"], columns="label", values="crps")
    rows = []
    for c in piv.columns:
        if c in ("ar1", "linear", "locf"):
            continue
        d = (piv[c] - piv["ar1"]).dropna()
        rows.append({"model": c, "cells": len(d), "lower_crps_than_ar1": float((d < 0).mean()),
                     "median_crps_diff": float(d.median())})
    emit(md(pd.DataFrame(rows), {"cells": "{:d}", "lower_crps_than_ar1": "{:.2f}",
                                 "median_crps_diff": "{:+.4f}"}))
    # figure
    stations = sorted(g.station.unique())
    rates = sorted(g.rate.unique())
    fig, axes = plt.subplots(len(stations), len(rates), figsize=(4.2 * len(rates), 2.8 * len(stations)),
                             squeeze=False, sharey=True)
    for i, st in enumerate(stations):
        for j, rt in enumerate(rates):
            ax = axes[i, j]
            h = g[(g.station == st) & (g.rate == rt)]
            labels = sorted(h.label.unique(), key=lambda s: (s in ("ar1", "linear", "locf"), s))
            for k, lb in enumerate(labels):
                q = h[h.label == lb].sort_values("block")
                col = (METHOD_COLOR.get(lb) or (METHOD_COLOR["hmc"] if lb.startswith("hmc") else C[0 if k == 0 else 4]))
                ax.plot(q.block, q.crps, marker="o", ms=4, lw=1.6, color=col, label=lb)
            ax.set_xscale("log")
            ax.set_xticks(sorted(h.block.unique()), [str(b) for b in sorted(h.block.unique())])
            ax.set_title(f"{st}, {int(rt * 100)} % hidden", fontsize=9, color=INK)
            ax.set_xlabel("block length (h)", fontsize=8, color=INK2)
            style(ax)
        axes[i, 0].set_ylabel("CRPS (log PM2.5)", fontsize=8, color=INK2)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "beijing_cv_crps.png", dpi=130)
    plt.close(fig)


def beijing_rawlog(fits: pd.DataFrame) -> None:
    section("Beijing: likelihood degeneracy on the undithered log series")
    f = fits[fits.series.isin(["huairou", "huairou_rawlog"]) & (fits.strategy == "available")
             & fits.model.isin(["hmc_in", "pmc_state"])]
    b = best_start(f, ["series", "K", "model"])
    b["min_state_sd"] = b.sd_sorted.map(lambda s: min(float(v) for v in str(s).split("|")))
    emit(md(b[["series", "K", "model", "ll", "bic", "mean_sorted", "sd_sorted", "min_state_sd", "copulas"]],
            {"K": "{:d}", "ll": "{:.1f}", "bic": "{:.1f}", "min_state_sd": "{:.2e}"}))


def pamap2_missing(miss: pd.DataFrame, figdir: Path) -> None:
    section("PAMAP2: state-dependent missingness")
    emit("Per K = 3 group: share of 100 Hz hand samples that are NaN, share of 2 Hz windows "
         "declared missing (≥ 5 of 50 samples NaN: the rule used; any NaN: sensitivity).")
    emit("")
    k3 = miss[miss.level == "K3"]
    cols = ["subject", "name", "n_windows", "sample_nan_pct", "win_missing_pct_ge10", "win_missing_pct_any"]
    emit(md(k3[cols], {"subject": "{:d}", "n_windows": "{:d}", "sample_nan_pct": "{:.2f}",
                       "win_missing_pct_ge10": "{:.2f}", "win_missing_pct_any": "{:.2f}"}))
    emit("")
    act = miss[miss.level == "activity"]
    p = act.pivot_table(index="name", columns="subject", values="win_missing_pct_ge10").reset_index()
    emit("Windows missing (≥ 10 % rule, %) per activity:")
    emit("")
    emit(md(p, {c: "{:.2f}" for c in p.columns if c != "name"}))
    emit("")
    ser = miss[miss.level == "series"][["subject", "n_windows", "sample_nan_pct", "win_missing_pct_ge10",
                                         "win_missing_pct_any", "mcar_pct"]]
    emit(md(ser, {"subject": "{:d}", "n_windows": "{:d}", "sample_nan_pct": "{:.3f}",
                  "win_missing_pct_ge10": "{:.2f}", "win_missing_pct_any": "{:.2f}", "mcar_pct": "{:.2f}"}))
    fig, ax = plt.subplots(figsize=(6.5, 3))
    names = ["rest", "locomotion", "vigorous", "transient"]
    subs = sorted(k3.subject.unique())
    w = 0.8 / len(subs)
    for k, s in enumerate(subs):
        vals = [k3[(k3.subject == s) & (k3.name == n)].win_missing_pct_ge10.sum() for n in names]
        ax.bar(np.arange(len(names)) + (k - (len(subs) - 1) / 2) * w, vals, w * 0.9, color=C[k],
               label=f"subject {s}")
    ax.set_xticks(np.arange(len(names)), names)
    ax.set_ylabel("windows missing (%)", fontsize=8, color=INK2)
    ax.set_title("PAMAP2 hand IMU: missing 2 Hz windows by activity group", fontsize=10, color=INK)
    style(ax)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "pamap2_missing_by_group.png", dpi=130)
    plt.close(fig)


def pamap2_classification(fits: pd.DataFrame, cls: pd.DataFrame, orc: pd.DataFrame, figdir: Path) -> None:
    section("PAMAP2: classification error (%, non-transient windows)")
    emit("`complete`: error on the complete series. Gapped series (real gaps + 10 % MCAR blocks of "
         "10 s): `marg` exact marginalisation; `plugin` / `linear` / `locf` fill-in then classify; "
         "errors on all / missing / observed windows, and on the real-gap and MCAR windows "
         "(marginalisation). Unsupervised rows: best-likelihood start; the gapped-series model "
         "is fitted on the gapped series (ICE \"available\" unless stated).")
    emit("")
    ok = fits[(fits.status == "ok") & fits.series.str.startswith("pamap2_")]
    best = best_start(ok, ["series", "K", "model", "strategy"])
    rows = []

    def pick(df, **kw):
        q = df
        for k, v in kw.items():
            q = q[q[k] == v]
        return q

    for subj in sorted(cls.subject.unique()):
        for K in sorted(cls.K.unique()):
            for model in ("hmc_in", "pmc_state", "pmc_pair"):
                o = pick(orc, subject=subj, K=K, model=model)
                if len(o):
                    r = {"subject": subj, "K": K, "model": model, "fit": "oracle"}
                    r["complete"] = 100 * pick(o, method="complete").err_all.iloc[0]
                    mg = pick(o, method="marginalise").iloc[0]
                    r.update(marg_all=100 * mg.err_all, marg_missing=100 * mg.err_missing,
                             marg_observed=100 * mg.err_observed, marg_real=100 * mg.err_real,
                             marg_mcar=100 * mg.err_mcar)
                    for meth in ("plugin", "linear", "locf"):
                        r[f"{meth}_missing"] = 100 * pick(o, method=meth).err_missing.iloc[0]
                    rows.append(r)
                for strat in ("available", "impute", "sem"):
                    bc = pick(best, series=f"pamap2_{subj}_complete", K=K, model=model)
                    bg = pick(best, series=f"pamap2_{subj}_gapped", K=K, model=model, strategy=strat)
                    if bg.empty or bc.empty:
                        continue
                    rc_ = pick(cls, series=f"pamap2_{subj}_complete", K=K, model=model, start=int(bc.start.iloc[0]))
                    rg = pick(cls, series=f"pamap2_{subj}_gapped", K=K, model=model, strategy=strat,
                              start=int(bg.start.iloc[0]))
                    r = {"subject": subj, "K": K, "model": model, "fit": f"unsup. {strat}"}
                    r["complete"] = 100 * pick(rc_, method="complete").err_all.iloc[0]
                    r["gapfit_on_complete"] = 100 * pick(rg, applied_to="complete").err_all.iloc[0]
                    mg = pick(rg, method="marginalise").iloc[0]
                    r.update(marg_all=100 * mg.err_all, marg_missing=100 * mg.err_missing,
                             marg_observed=100 * mg.err_observed, marg_real=100 * mg.err_real,
                             marg_mcar=100 * mg.err_mcar)
                    for meth in ("plugin", "linear", "locf"):
                        r[f"{meth}_missing"] = 100 * pick(rg, method=meth).err_missing.iloc[0]
                    rows.append(r)
    t = pd.DataFrame(rows)
    cols = ["subject", "K", "model", "fit", "complete", "gapfit_on_complete", "marg_all", "marg_missing",
            "marg_observed", "marg_real", "marg_mcar", "plugin_missing", "linear_missing", "locf_missing"]
    t = t[[c for c in cols if c in t.columns]]
    emit(md(t, {"subject": "{:d}", "K": "{:d}", **{c: "{:.1f}" for c in cols[4:]}}))
    emit("")
    n = cls[(cls.method == "marginalise")].drop_duplicates("subject")[["subject", "n_all", "n_missing", "n_real", "n_mcar"]]
    emit("Scored windows: " + "; ".join(
        f"subject {int(r.subject)}: {int(r.n_all)} (missing {int(r.n_missing)}: real {int(r.n_real)}, "
        f"MCAR {int(r.n_mcar)})" for r in n.itertuples()))
    emit("")
    # compact views
    emit("**Complete series: model error (oracle) vs estimation error (unsupervised, best start)**, "
         "error %; `bic`: the model with the lowest BIC on the complete series:")
    emit("")
    comp = t[t.fit.isin(["oracle", "unsup. available"])].copy()
    rows = []
    bic_best = best_start(fits[fits.series.str.endswith("_complete") & (fits.status == "ok")],
                          ["series", "K", "model"])
    for (subj, K), h in comp.groupby(["subject", "K"]):
        r = {"subject": subj, "K": K}
        for m in ("hmc_in", "pmc_state", "pmc_pair"):
            q = h[h.model == m]
            o, u = q[q.fit == "oracle"].complete, q[q.fit != "oracle"].complete
            r[f"oracle_{m}"] = o.iloc[0] if len(o) else np.nan
            r[f"unsup_{m}"] = u.iloc[0] if len(u) else np.nan
        bb = bic_best[(bic_best.series == f"pamap2_{subj}_complete") & (bic_best.K == K)]
        r["bic_model"] = bb.loc[bb.bic.idxmin(), "model"]
        r["unsup_bic"] = r.get(f"unsup_{r['bic_model']}", np.nan)
        rows.append(r)
    ct = pd.DataFrame(rows)
    emit(md(ct, {"subject": "{:d}", "K": "{:d}", **{c: "{:.1f}" for c in ct.columns if c.startswith(("oracle", "unsup"))}}))
    emit("")
    emit("**Gapped series, error on the missing windows** (mean over the three subjects, %): exact "
         "marginalisation vs fill-in-then-classify, and marginalisation on real-gap vs MCAR windows:")
    emit("")
    gm = t[t.fit.isin(["oracle", "unsup. available"])].groupby(["K", "fit", "model"])[
        ["marg_missing", "plugin_missing", "linear_missing", "locf_missing", "marg_real", "marg_mcar",
         "marg_observed"]].mean().reset_index()
    emit(md(gm, {"K": "{:d}", **{c: "{:.1f}" for c in gm.columns if c not in ("K", "fit", "model")}}))
    emit("")
    emit("**Complete-series fit vs gapped-series fit** (unsupervised, ICE \"available\", best start; "
         "error % on all scored windows): same model applied to the complete series, and the "
         "gapped fit with marginalisation on the gapped series:")
    emit("")
    cg = t[t.fit == "unsup. available"][["subject", "K", "model", "complete", "gapfit_on_complete", "marg_all"]]
    emit(md(cg, {"subject": "{:d}", "K": "{:d}", "complete": "{:.1f}", "gapfit_on_complete": "{:.1f}",
                 "marg_all": "{:.1f}"}))
    emit("")
    diff = (cg.gapfit_on_complete - cg.complete).abs()
    emit(f"|gapped-fit − complete-fit| error on the complete series: median {diff.median():.1f} points, "
         f"> 5 points in {int((diff > 5).sum())} of {len(diff)} (subject, K, model) cells.")
    # figure: per K, error on the missing windows by method (oracle and unsupervised), mean over subjects
    Ks = sorted(gm.K.unique())
    fig, axes = plt.subplots(len(Ks), 2, figsize=(10, 2.8 * len(Ks)), sharey="row", squeeze=False)
    cats = ["marg_missing", "plugin_missing", "linear_missing", "locf_missing"]
    names = ["marginalise", "plug-in mean", "linear", "LOCF"]
    models = [m for m in ("hmc_in", "pmc_state", "pmc_pair") if m in set(gm.model)]
    for i, K in enumerate(Ks):
        for j, fit in enumerate(("oracle", "unsup. available")):
            ax = axes[i, j]
            w = 0.8 / len(models)
            for k, m in enumerate(models):
                q = gm[(gm.K == K) & (gm.fit == fit) & (gm.model == m)]
                if q.empty:
                    continue
                ax.bar(np.arange(len(cats)) + (k - (len(models) - 1) / 2) * w, [q[c].iloc[0] for c in cats],
                       w * 0.9, color=C[[6, 0, 1][k]], label=m)
            ax.set_xticks(np.arange(len(cats)), names, fontsize=8)
            ax.set_title(f"K = {K}, {'supervised oracle' if fit == 'oracle' else 'unsupervised (ICE available)'}",
                         fontsize=9, color=INK)
            style(ax)
        axes[i, 0].set_ylabel("error on missing windows (%)", fontsize=8, color=INK2)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("PAMAP2, gapped series: classification of the missing windows (mean of 3 subjects)",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(figdir / "pamap2_errors.png", dpi=130)
    plt.close(fig)


def pamap2_features(fc: pd.DataFrame) -> None:
    section("PAMAP2: choice of the feature and of the K = 3 grouping")
    emit("Supervised HMC-IN error (%) on the complete 2 Hz series. log_sd: log-SD of the hand "
         "‖a‖ per window (used); mean: window mean of ‖a‖. K3: ascending stairs = locomotion, "
         "Nordic walking = vigorous (used); K3_alt: the reverse.")
    emit("")
    p = fc.assign(err=100 * fc.err).pivot_table(index=["feature", "grouping"], columns="subject",
                                                values="err").reset_index()
    p.columns = [str(c) for c in p.columns]
    emit(md(p, {c: "{:.1f}" for c in p.columns if c not in ("feature", "grouping")}))


def pamap2_mnar(mnar: pd.DataFrame) -> None:
    section("PAMAP2: ignorable vs missingness-aware oracle (HMC-IN), real gaps only")
    emit("Supervised HMC-IN; `aware` adds P(window missing | state) estimated from the labels. "
         "Errors (%) on the real-gap windows and on the observed windows; shares of the groups "
         "among the real-gap windows (truth) and the mean posterior at those windows.")
    emit("")
    m = mnar.copy()
    m["err_missing"] = 100 * m.err_missing
    m["err_observed"] = 100 * m.err_observed
    cols = ["subject", "K", "rule", "model", "n_missing", "err_missing", "err_observed", "miss_rate_by_state",
            "truth_share_missing", "posterior_share_missing", "max_abs_diff_vs_library"]
    emit(md(m[cols], {"subject": "{:d}", "K": "{:d}", "n_missing": "{:d}", "err_missing": "{:.1f}",
                      "err_observed": "{:.2f}", "max_abs_diff_vs_library": "{:.1e}"}))


def pamap2_strip_fig(strip: pd.DataFrame, figdir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.0), sharex=True, gridspec_kw={"height_ratios": [2, 1.3]})
    t = strip.t_s.to_numpy()
    ax = axes[0]
    ax.plot(t, strip.Y, color=INK2, lw=0.6)
    miss = (strip.real + strip.mcar).to_numpy() > 0
    ymin, ymax = strip.Y.min(), strip.Y.max()
    ax.fill_between(t, ymin, ymax, where=strip.mcar.to_numpy() > 0, color=C[3], alpha=0.25, lw=0, label="added MCAR")
    ax.scatter(t[strip.real.to_numpy() > 0], np.full(int(strip.real.sum()), ymax), marker="|", color=C[7],
               s=40, label="real gap")
    ax.set_ylabel("log SD hand ‖a‖", fontsize=8, color=INK2)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="lower right")
    style(ax)
    ax.set_title(f"PAMAP2 subject strip: {strip.model.iloc[0]} (states aligned to the groups)",
                 fontsize=10, color=INK)
    ax = axes[1]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#ffffff", C[6], C[0], C[1], INK])   # -1 transient, 0, 1, 2, missing
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    for k, col in enumerate(("group", "x_complete", "x_gapped")):
        v = strip[col].to_numpy().astype(float)[None, :] + 1
        ax.imshow(v, cmap=cmap, vmin=-0.5, vmax=4.5, aspect="auto", interpolation="nearest",
                  extent=(t[0] - dt / 2, t[-1] + dt / 2, 2 - k - 0.4, 2 - k + 0.4))
    ax.imshow(np.where(miss, 4.0, 0.0)[None, :], cmap=cmap, vmin=-0.5, vmax=4.5, aspect="auto",
              interpolation="nearest", extent=(t[0] - dt / 2, t[-1] + dt / 2, -0.62, -0.45))
    ax.set_ylim(-0.75, 2.5)
    ax.set_xlim(t[0] - dt / 2, t[-1] + dt / 2)
    ax.set_yticks([2, 1, 0], ["truth", "complete fit", "gapped fit"], fontsize=7)
    from matplotlib.patches import Patch
    K = int(str(strip.model.iloc[0]).split("K=")[1][0])
    names = ("rest", "active") if K == 2 else ("rest", "locomotion", "vigorous")
    handles = [Patch(color=C[[6, 0, 1][g]], label=n) for g, n in enumerate(names)]
    handles.append(Patch(color=INK, label="missing window (black ticks)"))
    ax.legend(handles=handles, fontsize=7, frameon=False, ncol=len(handles), loc="upper center",
              bbox_to_anchor=(0.5, -0.45))
    ax.set_xlabel("time (s)", fontsize=8, color=INK2)
    style(ax)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(figdir / "pamap2_strip.png", dpi=130)
    plt.close(fig)


def forecast_table(fc: pd.DataFrame) -> None:
    section("Aotizhongxin: rolling forecasts (last 25 % of the year, origins every 24 h)")
    hs = [h for h in (1, 2, 3, 6, 12, 24) if h in set(fc.h)]
    f = fc[fc.h.isin(hs)]
    for metric in ("rmse", "crps_gauss", "cov90"):
        if metric not in f.columns:
            continue
        p = f.pivot_table(index="method", columns="h", values=metric).reset_index()
        emit(f"**{metric}** by horizon (h, hours)" + (" — Gaussian CRPS from the predictive mean and sd"
                                                     if metric == "crps_gauss" else ""))
        emit("")
        emit(md(p, {c: "{:.3f}" for c in p.columns if c != "method"}))
        emit("")
    emit(f"Origins: {int(fc.n.max())}.")


def strategy_fig(t: pd.DataFrame, fits: pd.DataFrame, figdir: Path) -> None:
    ok = fits[(fits.status == "ok") & fits.series.isin(["tsnh4", "huairou", "pamap2_102_gapped"])]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, s in zip(axes, ["tsnh4", "huairou", "pamap2_102_gapped"]):
        g = ok[ok.series == s]
        if g.empty:
            ax.set_visible(False)
            continue
        for strat in ("available", "impute", "sem"):
            h = g[g.strategy == strat]
            gap = h.ll_gap_to_best.clip(lower=0.1)
            ax.scatter(h.runtime_s, gap, s=16, color=STRAT_COLOR[strat], label=strat, alpha=0.8,
                       edgecolor="white", linewidth=0.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("estimation time (s)", fontsize=8, color=INK2)
        ax.set_title(s, fontsize=9, color=INK)
        style(ax)
    axes[0].set_ylabel("LL below best fit of the cell (nats, floor 0.1)", fontsize=8, color=INK2)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "strategies_ll_vs_time.png", dpi=130)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--figures", default=str(HERE / "figures"))
    ap.add_argument("--extracts", default=str(HERE.parent / "out/real_series/full/figure_data"),
                    help="data extracts of the two time-series figures (written by the runner "
                         "outside the repository: they contain observations)")
    args = ap.parse_args(argv)
    res, figdir, ext = Path(args.results), Path(args.figures), Path(args.extracts)
    figdir.mkdir(parents=True, exist_ok=True)

    def read(name):
        p = res / f"{name}.csv"
        if not p.exists():
            p = ext / f"{name}.csv"
        if not p.exists() and name in ("tsnh4_longest_gap", "pamap2_strip"):
            print(f"[summarise] {p} not found: figure skipped (run the campaign first).")
        return pd.read_csv(p) if p.exists() else None

    fits = read("fits")
    info = json.loads((res / "run_info.json").read_text())
    emit("# A4 real-series study: tables")
    emit("")
    emit("Generated by `report/real_series/summarise.py` from the CSVs of this folder.")
    overview(fits, read("missingness_pamap2"), info)
    failures(fits, read("task_errors"))
    bic_selection(fits)
    t = strategies(fits)
    strategy_fig(t, fits, figdir)
    if read("imputation_tsnh4") is not None:
        tsnh4_imputation(fits, read("imputation_tsnh4"), figdir)
    if read("tsnh4_longest_gap") is not None:
        tsnh4_zoom_fig(read("tsnh4_longest_gap"), figdir)
    if read("cv_beijing") is not None:
        beijing_cv(read("cv_beijing"), fits, figdir)
    if (fits.series == "huairou_rawlog").any():
        beijing_rawlog(fits)
    if read("forecast_beijing") is not None:
        forecast_table(read("forecast_beijing"))
    if read("pamap2_feature_check") is not None:
        pamap2_features(read("pamap2_feature_check"))
    if read("missingness_pamap2") is not None:
        pamap2_missing(read("missingness_pamap2"), figdir)
    if read("classification_pamap2") is not None:
        pamap2_classification(fits, read("classification_pamap2"), read("oracle_pamap2"), figdir)
    if read("mnar_pamap2") is not None:
        pamap2_mnar(read("mnar_pamap2"))
    if read("pamap2_strip") is not None:
        pamap2_strip_fig(read("pamap2_strip"), figdir)
    runtime_table(fits, read("tasks"))
    (res / "tables.md").write_text("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
