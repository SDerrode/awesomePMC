"""
summarise.py — tables and figures of the PAMAP2 study of state-dependent
missingness (P6), from the CSVs of ``run_pamap2.py`` only (no estimation).

Prints every table of ``README.md`` (Markdown), writes them to
``results/tables.md``, a few of them as LaTeX (``tables/*.tex``, booktabs),
and the figures to ``figures/``.

Usage (repository root)::

    .venv/bin/python report/missing_state/pamap2/summarise.py
    .venv/bin/python report/missing_state/pamap2/summarise.py --results report/missing_state/pamap2/results_quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pm_common as pc  # noqa: E402

# Palette of the real-series study (categorical slots in fixed order, inks).
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dddcd6"
SUBJ_COLOR = {102: C[0], 108: C[1], 105: C[2]}

OUT: list[str] = []
TEX: dict[str, str] = {}


def emit(text: str = "") -> None:
    print(text)
    OUT.append(text)


def _cell(v, f) -> str:
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "–"
        return f(v) if callable(f) else f.format(v)
    return str(v)


def md(df: pd.DataFrame, fmt: dict | None = None, default: str = "{:.3f}") -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_cell(r[c], fmt.get(c, default)) for c in cols) + " |")
    return "\n".join(lines)


def tex(name: str, df: pd.DataFrame, fmt: dict | None = None, default: str = "{:.3f}",
        caption: str = "") -> None:
    fmt = fmt or {}
    cols = list(df.columns)

    def esc(s):
        return (str(s).replace("_", r"\_").replace("%", r"\%").replace("≥", r"$\geq$")
                .replace("π", r"$\pi$").replace("χ²", r"$\chi^2$").replace("–", "--")
                .replace("→", r"$\to$").replace("±", r"$\pm$"))
    lines = [r"\begin{table}[ht]", r"\centering", r"\small",
             r"\begin{tabular}{" + "l" * 2 + "r" * (len(cols) - 2) + "}", r"\toprule",
             " & ".join(esc(c) for c in cols) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        lines.append(" & ".join(esc(_cell(r[c], fmt.get(c, default))) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if caption:
        lines.append(r"\caption{" + esc(caption) + "}")
    lines.append(r"\end{table}")
    TEX[name] = "\n".join(lines) + "\n"


def pct(v) -> str:
    return f"{100 * v:.2f}"


def vec(s, scale=1.0, f="{:.3f}") -> str:
    if not isinstance(s, str) or not s:
        return "–"
    return " / ".join(f.format(scale * float(x)) for x in s.split("|"))


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


GROUP_LABEL = {("K3", -1): "transient", ("K3", 0): "rest", ("K3", 1): "locomotion",
               ("K3", 2): "vigorous", ("K2", -1): "transient", ("K2", 0): "rest",
               ("K2", 1): "active"}


# ---------------------------------------------------------------------------
# Part 1
# ---------------------------------------------------------------------------

def part_describe(res: Path, fig: Path) -> None:
    h = pd.read_csv(res / "describe_100hz.csv")
    w = pd.read_csv(res / "describe_windows.csv")
    subjects = [s for s in pc.SUBJECTS if s in set(h.subject)]
    emit("## 1. Description of the real hand-IMU gaps\n")
    emit("### 1a. 100 Hz samples, per activity (missing %, onset a, persistence b, bursts)\n")
    a = h[h.level == "activity"].copy()
    a["activity"] = a.code.map(lambda c: pc.ACTIVITIES.get(int(c), "transient (0)"))
    rows = []
    for c in sorted(a.code.unique()):
        r = {"activity": a[a.code == c].activity.iloc[0]}
        for s in subjects:
            x = a[(a.code == c) & (a.subject == s)]
            if x.empty:
                continue
            x = x.iloc[0]
            r[f"{s} miss %"] = 100 * x.rate
            r[f"{s} bursts"] = x.n_bursts
            r[f"{s} b"] = x.persistence
        rows.append(r)
    t = pd.DataFrame(rows)
    fmt = {c: "{:.3f}" for c in t.columns if "miss" in c}
    fmt.update({c: "{:.0f}" for c in t.columns if "bursts" in c})
    fmt.update({c: "{:.2f}" for c in t.columns if c.endswith(" b")})
    emit(md(t, fmt))
    tex("describe_100hz_activity", t, fmt, caption="Hand-IMU dropouts at 100 Hz per activity")
    emit()
    # Stability across subjects: Spearman correlation of the per-activity rates
    from scipy.stats import spearmanr
    piv = a[a.code != 0].pivot(index="code", columns="subject", values="rate")
    emit("Spearman correlation of the per-activity 100 Hz missing rates (12 protocol activities):")
    pairs = [(s1, s2) for i, s1 in enumerate(subjects) for s2 in subjects[i + 1:]]
    emit(", ".join(f"{s1}–{s2}: {spearmanr(piv[s1], piv[s2]).statistic:.2f}" for s1, s2 in pairs))
    emit()
    emit("### 1b. 100 Hz, per K = 3 group\n")
    g = h[h.level == "K3"].copy()
    g["group"] = [GROUP_LABEL[("K3", int(c))] for c in g.code]
    cols = ["subject", "group", "n", "rate", "n_bursts", "burst_mean", "burst_max", "share_len1",
            "onset", "persistence", "b_over_pi", "dispersion"]
    t = g[cols].rename(columns={"rate": "miss %", "n_bursts": "bursts", "burst_mean": "mean len",
                                "burst_max": "max len", "share_len1": "share len 1",
                                "onset": "a", "persistence": "b", "b_over_pi": "b/π",
                                "dispersion": "disp. (missing per 1 s)"})
    t["miss %"] *= 100
    fmt = {"n": "{:.0f}", "miss %": "{:.3f}", "bursts": "{:.0f}", "mean len": "{:.2f}",
           "max len": "{:.0f}", "share len 1": "{:.2f}", "a": "{:.5f}", "b": "{:.3f}",
           "b/π": "{:.0f}", "disp. (missing per 1 s)": "{:.2f}"}
    emit(md(t, fmt))
    tex("describe_100hz_groups", t, fmt, caption="Hand-IMU dropouts at 100 Hz per K = 3 group")
    emit()
    emit("Onset dispersion (variance / mean of the burst onsets per 1 s and 10 s block inside an "
         "activity; 1 for memoryless onsets):\n")
    rows = []
    for s in subjects:
        x = a[(a.subject == s) & (a.code.isin([4, 5, 7, 24]))]
        for _, r in x.iterrows():
            rows.append({"subject": s, "activity": r.activity, "bursts": r.n_bursts,
                         "disp 1 s": r.onset_disp_1s, "disp 10 s": r.onset_disp_10s})
    emit(md(pd.DataFrame(rows), {"bursts": "{:.0f}", "disp 1 s": "{:.2f}", "disp 10 s": "{:.2f}"}))
    emit()
    emit("### 1c. 2 Hz windows (the model's series), per K = 3 group\n")
    for rule in pc.RULES:
        x = w[(w.rule == rule) & (w.level.isin(["K3", "series"]))].copy()
        x["group"] = [GROUP_LABEL.get((lv, int(c)), "whole series") for lv, c in zip(x.level, x.code)]
        cols = ["subject", "group", "n", "n_missing", "rate", "n_bursts", "burst_mean", "burst_max",
                "onset", "persistence", "b_over_pi"]
        t = x[cols].rename(columns={"n_missing": "M", "rate": "π", "n_bursts": "bursts",
                                    "burst_mean": "mean len", "burst_max": "max len",
                                    "onset": "a", "persistence": "b", "b_over_pi": "b/π"})
        fmt = {"n": "{:.0f}", "M": "{:.0f}", "π": "{:.4f}", "bursts": "{:.0f}",
               "mean len": "{:.2f}", "max len": "{:.0f}", "a": "{:.4f}", "b": "{:.3f}",
               "b/π": "{:.1f}"}
        emit(f"Rule `{rule}`:\n")
        emit(md(t, fmt))
        tex(f"describe_windows_{rule}", t, fmt, caption=f"Missing 2 Hz windows, rule {rule}")
        emit()
    # goodness of fit
    gof = pd.read_csv(res / "gof.csv")
    emit("### 1d. Goodness of fit of the mechanisms (masks redrawn on the true class path)\n")
    emit("Observed statistic, and median [2.5 %, 97.5 %] of the redrawn masks; p two-sided.\n")
    for (resol, classes), sub in gof.groupby(["resolution", "classes"], sort=False):
        emit(f"{resol}, classes = {classes}:\n")
        rows = []
        for (s, rule, mech), x in sub.groupby(["subject", "rule", "mechanism"], sort=False):
            r = {"subject": s, "rule": rule, "mechanism": mech}
            for _, y in x.iterrows():
                if y.statistic in ("M",):
                    continue
                r[y.statistic] = (f"{y.observed:.3g} ({y.sim_median:.3g} [{y.sim_lo:.3g}, "
                                  f"{y.sim_hi:.3g}]; p {y.p_two_sided:.2g})")
            rows.append(r)
        emit(md(pd.DataFrame(rows)))
        emit()
    fig_describe(h, w, subjects, fig)


def fig_describe(h, w, subjects, fig: Path) -> None:
    a = h[(h.level == "activity") & (h.code != 0)]
    order = [1, 2, 3, 17, 6, 13, 12, 16, 4, 7, 5, 24]
    names = [pc.ACTIVITIES[c] for c in order]
    fig_, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    y = np.arange(len(order))
    for k, s in enumerate(subjects):
        v = [100 * a[(a.subject == s) & (a.code == c)].rate.iloc[0] for c in order]
        ax.scatter(np.maximum(v, 1e-3), y + (k - 1) * 0.22, s=22, color=SUBJ_COLOR[s],
                   label=f"subject {s}", zorder=3)
    ax.set_xscale("log")
    ax.set_yticks(y, names)
    for yy in (3.5, 8.5):
        ax.axhline(yy, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("hand samples missing at 100 Hz (%, log scale; 0 drawn at 0.001)",
                  color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Dropout rate per activity (dotted: rest | locomotion | vigorous)", fontsize=10,
                 color=INK, loc="left")
    style(ax)
    ax = axes[1]
    x = w[(w.rule == "any") & (w.level == "K3") & (w.code >= 0)]
    width = 0.25
    for k, s in enumerate(subjects):
        xs = x[x.subject == s].sort_values("code")
        pos = np.arange(3) + (k - 1) * width
        ax.bar(pos - width / 4, xs.onset, width / 2, color=SUBJ_COLOR[s], label=f"{s}: onset a")
        ax.bar(pos + width / 4, xs.persistence, width / 2, color=SUBJ_COLOR[s], alpha=0.45,
               label=f"{s}: persistence b")
    ax.set_xticks(np.arange(3), pc.GROUP_NAMES[3])
    ax.set_ylabel("probability", color=INK2, fontsize=9)
    ax.set_title("2 Hz windows, rule 'any': onset vs persistence", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper left")
    style(ax)
    fig_.tight_layout()
    fig_.savefig(fig / "gaps_by_activity.png", dpi=130)
    plt.close(fig_)


def part_supervised(res: Path) -> None:
    m = pd.read_csv(res / "supervised_mechanism.csv")
    lr = pd.read_csv(res / "supervised_lr.csv")
    emit("## 2a. Supervised view: mechanisms fitted on the true groups (plain MLE, no guard)\n")
    emit("Groups in order rest / (locomotion /) vigorous or rest / active; LR (df) of the label-"
         "based tests — the Bernoulli ones ignore the bursts.\n")
    rows = []
    for (s, rule, K), x in m[m.estimate == "plain"].groupby(["subject", "rule", "K"], sort=False):
        st = x[x.mechanism == "state"].iloc[0]
        sm = x[x.mechanism == "state-markov"].iloc[0]
        r = {"subject": s, "rule": rule, "K": K, "π (%)": vec(st.rates, 100, "{:.2f}"),
             "a (%)": vec(sm.onset, 100, "{:.2f}"), "b": vec(sm.persistence, 1, "{:.2f}")}
        for alt, null in (("state", "common"), ("state-markov", "common"), ("state-markov", "state")):
            y = lr[(lr.subject == s) & (lr.rule == rule) & (lr.K == K) & (lr.alternative == alt)
                   & (lr.null == null)].iloc[0]
            r[f"LR {alt} vs {null}"] = f"{y.LR:.0f} ({int(y.df)})"
        rows.append(r)
    t = pd.DataFrame(rows)
    emit(md(t))
    tex("supervised_mechanism", t, caption="Mechanisms fitted on the true groups")
    emit()


# ---------------------------------------------------------------------------
# Part 2
# ---------------------------------------------------------------------------

def part_tests(res: Path) -> None:
    """``lr_tests.csv``: the profile statistic (library 0758e3a), χ² for every
    cell, bootstrap where B > 0; ``lr_tests_fits_statistic.csv`` (optional):
    the first run, statistic = difference of the two ICE fixed points, with
    its bootstrap (B = 99 rule ge10, 19 rule any)."""
    t = pd.read_csv(res / "lr_tests.csv")
    key = ["rule", "K", "subject", "alternative", "null"]
    old_path = res / "lr_tests_fits_statistic.csv"
    old = pd.read_csv(old_path).set_index(key) if old_path.exists() else None
    w = pd.read_csv(res / "describe_windows.csv")
    bursts = {(r.subject, r.rule): r.n_bursts for r in w[w.level == "series"].itertuples()}
    emit("## 2. Likelihood-ratio tests on the real series (unsupervised HMC-IN)\n")
    emit("LR: the profile statistic of `missingness_lr_test` (library commit 0758e3a); "
         "LR fits: the difference of the two ICE fixed points (`statistic_fits`). "
         "p boot: parametric bootstrap of the profile statistic (B = 99, subject 108, rule "
         "ge10); p boot (fits): the first run's bootstrap of the fits' statistic (B = 99 rule "
         "ge10, 19 rule any). Bursts: runs of missing windows in the series.\n")
    rows = []
    t = t.sort_values(key, key=lambda c: c.map({"ge10": 0, "any": 1}) if c.name == "rule" else c)
    for _, r in t.iterrows():
        o = old.loc[tuple(r[k] for k in key)] if old is not None else None
        rows.append({"rule": r.rule, "K": r.K, "subject": r.subject, "M": r.M,
                     "bursts": bursts.get((r.subject, r.rule), np.nan),
                     "test": f"{r.alternative} vs {r.null}", "LR": r.LR,
                     "LR fits": r.get("LR_fits", np.nan), "df": r.df, "p χ²": r.p_chi2,
                     "p boot": r.p_boot if r.B else np.nan,
                     "boot q95 / max": (f"{r.boot_q95:.2f} / {r.boot_max:.2f}" if r.B else ""),
                     "p boot (fits)": o.p_boot if o is not None else np.nan,
                     "B (fits)": int(o.B_valid) if o is not None else np.nan,
                     "align err": r.align_err})
    fmt = {"M": "{:.0f}", "bursts": "{:.0f}", "LR": "{:.1f}", "LR fits": "{:.1f}", "df": "{:.0f}",
           "p χ²": "{:.1e}", "p boot": "{:.3f}", "p boot (fits)": "{:.3f}", "B (fits)": "{:.0f}",
           "align err": "{:.3f}"}
    tt = pd.DataFrame(rows)
    emit(md(tt, fmt))
    tex("lr_tests", tt, fmt, caption="Likelihood-ratio tests of state-dependent missingness")
    emit()
    if old is not None:
        o = old.reset_index()
        from scipy.stats import chi2
        emit("Bootstrap of the fits' statistic (first run) against χ²: mean, 95 % quantile "
             "and maximum of the replicates, per test and df.\n")
        rows = []
        for (alt, null, df), x in o.groupby(["alternative", "null", "df"]):
            rows.append({"test": f"{alt} vs {null}", "df": df, "cells": len(x),
                         "boot mean": f"{x.boot_mean.min():.2f}–{x.boot_mean.max():.2f}",
                         "boot q95": f"{x.boot_q95.min():.2f}–{x.boot_q95.max():.2f}",
                         "χ² q95": f"{chi2.ppf(0.95, df):.2f}",
                         "boot max": f"{x.boot_max.max():.2f}"})
        emit(md(pd.DataFrame(rows), {"df": "{:.0f}", "cells": "{:.0f}"}))
        emit()
    emit("Fitted mechanisms (H1; states sorted by margin mean, i.e. by hand-motion intensity):\n")
    rows = []
    for _, r in t.iterrows():
        if r.alternative == "state":
            par = f"π = {vec(r.alt_rates, 100, '{:.2f}')} %"
        else:
            par = f"a = {vec(r.alt_onset, 100, '{:.2f}')} %, b = {vec(r.alt_persistence, 1, '{:.2f}')}"
        rows.append({"rule": r.rule, "K": r.K, "subject": r.subject,
                     "test": f"{r.alternative} vs {r.null}", "H1 estimate": par,
                     "state means": vec(r.alt_means, 1, "{:.2f}"),
                     "state composition (rest/…)": r.state_composition})
    emit(md(pd.DataFrame(rows)))
    emit()


# ---------------------------------------------------------------------------
# Parts 3–4
# ---------------------------------------------------------------------------

VARIANTS = ("ignorable", "state", "state-markov", "state (warm)", "state-markov (warm)")
VORDER = {v: i for i, v in enumerate(VARIANTS)}


def load_cls(res: Path) -> pd.DataFrame:
    """classification.csv with a ``variant``: the mechanism, "(warm)" when it was
    estimated from the ignorable fit of the same start (the LR test's protocol)."""
    c = pd.read_csv(res / "classification.csv")
    if "start_from" not in c:
        c["start_from"] = np.nan
    warm = c.start_from == "ignorable fit"
    c["variant"] = np.where(warm, c.mechanism + " (warm)", c.mechanism)
    for col in ("mask", "r"):
        if col not in c:
            c[col] = np.nan
    return c


def _best_start(c: pd.DataFrame) -> pd.DataFrame:
    """Unsupervised: the start with the highest log-likelihood of its own model."""
    u = c[c.fit == "unsupervised"]
    keys = ["setting", "subject", "K", "rule", "model", "variant", "mask", "r"]
    idx = u.groupby(keys, dropna=False)["ll"].idxmax()
    return pd.concat([c[c.fit == "oracle"], u.loc[idx]], ignore_index=True)


def _sort(t: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    def key(s):
        if s.name in ("mechanism", "variant"):
            return s.map(VORDER)
        if s.name == "rule":
            return s.map({"ge10": 0, "any": 1})
        return s
    return t.sort_values(keys, key=key)


METRICS = {"err_all": "err all %", "err_missing": "err missing %", "acc_missing": "acc miss.",
           "conf_missing": "conf miss.", "brier_missing": "Brier miss.",
           "logloss_missing": "log-loss miss.", "ece_missing": "ECE miss."}
CLS_FMT = {"M": "{:.0f}", "n": "{:.0f}", "err all %": "{:.1f}", "err missing %": "{:.1f}",
           "acc miss.": "{:.3f}", "conf miss.": "{:.3f}", "Brier miss.": "{:.3f}",
           "log-loss miss.": "{:.3f}", "ECE miss.": "{:.3f}"}


def _paired(grp: pd.DataFrame, metric: str, variant: str) -> tuple[float, float]:
    """Mean and standard error over (subject, r) of metric(variant) − metric(ignorable)."""
    piv = grp.pivot_table(index=["subject", "r"], columns="variant", values=metric, dropna=False)
    if variant not in piv or "ignorable" not in piv:
        return np.nan, np.nan
    d = (piv[variant] - piv["ignorable"]).dropna()
    return d.mean(), (d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan)


def _cls_table(c: pd.DataFrame, keys: list[str], paired: bool) -> pd.DataFrame:
    rows = []
    for key, x in c.groupby(keys, sort=False, dropna=False):
        r = dict(zip(keys, key))
        r["M"] = x.n_missing.mean()
        r["n"] = len(x)
        for k, v in METRICS.items():
            r[v] = x[k].mean() * (100 if k.startswith("err") else 1)
        if paired:
            grp = c
            for kk, vv in zip(keys[:-1], key[:-1]):
                grp = grp[grp[kk] == vv]
            if key[-1] == "ignorable":
                r["Δ err miss. (pts)"] = r["Δ log-loss miss."] = ""
            else:
                m, s = _paired(grp.assign(r=grp.r.fillna(0)), "err_missing", key[-1])
                r["Δ err miss. (pts)"] = f"{100 * m:+.2f} ± {100 * s:.2f}"
                m, s = _paired(grp.assign(r=grp.r.fillna(0)), "logloss_missing", key[-1])
                r["Δ log-loss miss."] = f"{m:+.3f} ± {s:.3f}"
        else:
            per = [x[x.subject == s].err_missing for s in pc.SUBJECTS]
            r["err miss. per subject"] = " / ".join(
                f"{100 * p.iloc[0]:.1f}" if len(p) else "–" for p in per)
        rows.append(r)
    return _sort(pd.DataFrame(rows), keys)


def part_real(res: Path) -> None:
    c = _best_start(load_cls(res))
    c = c[c.setting == "real"]
    emit("## 3. Classification — real gaps\n")
    emit("Mean over the three subjects; last column: error at the missing windows for "
         "102 / 108 / 105. Unsupervised HMC-IN: best of 3 starts by its own likelihood "
         "(\"(warm)\": the mechanism estimated from the ignorable fit of the same start); "
         "PMC pair: k-means start, warm variants only.\n")
    t = _cls_table(c, ["rule", "K", "fit", "model", "variant"], paired=False)
    emit(md(t.drop(columns=["n"]), CLS_FMT))
    tex("classification_real", t.drop(columns=["n", "acc miss.", "ECE miss."]), CLS_FMT,
        caption="Classification with the real gaps")
    emit()
    if "err_by_activity" in c:
        emit("Oracles, K = 3: MPM error per activity at the missing / observed windows (%), "
             "and the share of the activity's windows that are missing:\n")
        rows = []
        o = c[(c.fit == "oracle") & (c.K == 3) & c.err_by_activity.notna()]
        for _, r in _sort(o, ["rule", "model", "subject", "variant"]).iterrows():
            row = {"rule": r.rule, "model": r.model, "subject": r.subject, "variant": r.variant}
            for item in r.err_by_activity.split("|"):
                a_id, em, nm, eo, no = item.split(":")
                a_id, nm, no = int(a_id), int(nm), int(no)
                if a_id in (4, 5, 7, 24, 6, 12, 13, 16):
                    miss_share = nm / max(nm + no, 1)
                    em_s = "–" if em == "nan" else f"{100 * float(em):.0f}"
                    row[pc.ACTIVITIES[a_id]] = (f"{em_s} / {100 * float(eo):.0f} "
                                                f"({100 * miss_share:.0f} %)")
            rows.append(row)
        emit(md(pd.DataFrame(rows)))
        emit()
    u = c[(c.fit == "unsupervised") & (c.variant != "ignorable")]
    emit("Unsupervised fits: estimated mechanism (states sorted by margin mean) and share of the "
         "missing windows' posterior mass per aligned state vs the truth:\n")
    rows = []
    for _, r in _sort(u, ["rule", "K", "model", "subject", "variant"]).iterrows():
        par = (f"π = {vec(r.sorted_rates, 100, '{:.2f}')} %" if r.mechanism == "state" else
               f"a = {vec(r.sorted_onset, 100, '{:.2f}')} %, b = {vec(r.sorted_persistence, 1, '{:.2f}')}")
        rows.append({"rule": r.rule, "K": r.K, "model": r.model, "subject": r.subject,
                     "variant": r.variant, "estimate": par, "err all %": 100 * r.err_all,
                     "true share": vec(r.share_true_missing, 1, "{:.2f}"),
                     "posterior share": vec(r.share_post_missing, 1, "{:.2f}")})
    emit(md(pd.DataFrame(rows), {"err all %": "{:.1f}"}))
    emit()


def part_basins(res: Path) -> None:
    """Direct vs warm-started estimation of the mechanism: likelihood reached."""
    c = load_cls(res)
    u = c[(c.fit == "unsupervised")].copy()
    if u.empty or "ll_mask_common" not in u:
        return
    emit("## 3c. Estimating the mechanism from the start or from the ignorable fit\n")
    emit("Per (series, start): log p(y_obs, m) of the MNAR fit minus [log p(y_obs) of the "
         "ignorable fit + log p(m) of the common mechanism] — the value the MNAR fit starts "
         "from in the warm protocol, and a lower bound of its fixed point where ICE is EM. "
         "Negative: the fit ended in a worse basin than the ignorable fit.\n")
    keys = ["setting", "subject", "K", "rule", "model", "start", "mask", "r"]
    rows = []
    for key, x in u.groupby(keys, dropna=False):
        ig = x[x.variant == "ignorable"]
        if ig.empty:
            continue
        ig = ig.iloc[0]
        base = {m: ig.ll + float(v) for m, v in zip(("state", "state-markov"),
                                                     str(ig.ll_mask_common).split("|"))}
        for _, y in x[x.variant != "ignorable"].iterrows():
            rows.append({**dict(zip(keys, key)), "variant": y.variant,
                         "gain": y.ll - base[y.mechanism],
                         "d_err_all": y.err_all - ig.err_all,
                         "d_err_missing": y.err_missing - ig.err_missing})
    d = pd.DataFrame(rows)
    if d.empty:
        return
    out = []
    for key, x in d.groupby(["setting", "model", "K", "variant"], sort=False):
        out.append({"setting": key[0], "model": key[1], "K": key[2], "variant": key[3],
                    "fits": len(x), "median gain (nat)": x.gain.median(),
                    "share gain < −1": np.mean(x.gain < -1), "min gain": x.gain.min(),
                    "mean Δ err all (pts)": 100 * x.d_err_all.mean(),
                    "mean Δ err missing (pts)": 100 * x.d_err_missing.mean()})
    t = _sort(pd.DataFrame(out), ["setting", "model", "K", "variant"])
    fmt = {"fits": "{:.0f}", "median gain (nat)": "{:.1f}", "share gain < −1": "{:.2f}",
           "min gain": "{:.1f}", "mean Δ err all (pts)": "{:+.1f}",
           "mean Δ err missing (pts)": "{:+.1f}"}
    emit(md(t, fmt))
    tex("mechanism_start", t, fmt, caption="Direct vs warm-started estimation of the mechanism")
    emit()


def part_controlled(res: Path, fig: Path) -> None:
    c = load_cls(res)
    c = c[c.setting == "controlled"]
    if c.empty:
        return
    emit("## 3b. Classification — controlled masks (simulated on the true groups)\n")
    emit("Mean over subjects × mask seeds; Δ = paired difference to the ignorable fit of the same "
         "series (mean ± s.e.). Unsupervised HMC-IN: k-means start.\n")
    t = _cls_table(c, ["rule", "mask", "K", "fit", "model", "variant"], paired=True)
    emit(md(t.drop(columns=["acc miss.", "ECE miss."]), CLS_FMT))
    tex("classification_controlled", t.drop(columns=["acc miss.", "ECE miss.", "n"]), CLS_FMT,
        caption="Classification with masks simulated on the true groups")
    emit()
    calibration_table(res)
    fig_reliability(res, fig)
    fig_errors(res, fig)


VCOLOR = {"ignorable": C[6], "state": C[1], "state-markov": C[2], "state (warm)": C[3],
          "state-markov (warm)": C[0]}
ERR_FLOOR = 0.05   # % — zero errors are drawn at this floor on the log axes


def calibration_table(res: Path) -> None:
    """Mean predicted error (1 − max posterior) vs observed MPM error at the missing windows."""
    c = _best_start(load_cls(res))
    c = c[c.model == "hmc_in"]
    emit("### Calibration at the missing windows (HMC-IN, both K; mean over series)\n")
    emit("Predicted error = 1 − mean max posterior; observed = MPM error; the posteriors are "
         "almost all above 0.95, so a binned reliability diagram is one bin.\n")
    rows = []
    for key, x in c.groupby(["setting", "rule", "fit", "variant"], sort=False):
        rows.append({"setting": key[0], "rule": key[1], "fit": key[2], "variant": key[3],
                     "series": len(x), "predicted err %": 100 * (1 - x.conf_missing).mean(),
                     "observed err %": 100 * x.err_missing.mean(),
                     "ratio obs/pred": x.err_missing.mean() / max((1 - x.conf_missing).mean(), 1e-12),
                     "log-loss": x.logloss_missing.mean(), "Brier": x.brier_missing.mean()})
    t = _sort(pd.DataFrame(rows), ["setting", "rule", "fit", "variant"])
    fmt = {"series": "{:.0f}", "predicted err %": "{:.2f}", "observed err %": "{:.2f}",
           "ratio obs/pred": "{:.1f}", "log-loss": "{:.3f}", "Brier": "{:.3f}"}
    emit(md(t, fmt))
    tex("calibration", t, fmt, caption="Calibration at the missing windows")
    emit()


def fig_reliability(res: Path, fig: Path) -> None:
    """Predicted vs observed error at the missing windows, one point per series."""
    c = _best_start(load_cls(res))
    c = c[c.model == "hmc_in"]
    fig_, axes = plt.subplots(2, 2, figsize=(10.5, 8.6), sharex=True, sharey=True)
    marker = {"ge10": "o", "any": "s"}
    for i, fit in enumerate(("oracle", "unsupervised")):
        for j, setting in enumerate(("real", "controlled")):
            ax = axes[i, j]
            x = c[(c.fit == fit) & (c.setting == setting)]
            ax.plot([ERR_FLOOR, 100], [ERR_FLOOR, 100], color=INK2, lw=0.8, ls="--")
            for v in VARIANTS:
                for rule, mk in marker.items():
                    y = x[(x.variant == v) & (x.rule == rule)]
                    if y.empty:
                        continue
                    ax.scatter(np.maximum(100 * (1 - y.conf_missing), ERR_FLOOR),
                               np.maximum(100 * y.err_missing, ERR_FLOOR), s=18, marker=mk,
                               color=VCOLOR[v], alpha=0.75, edgecolor="white", linewidth=0.4,
                               label=f"{v}, {rule}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            where = "real gaps" if setting == "real" else "simulated masks"
            ax.set_title(f"{fit} HMC-IN, {where}", fontsize=10, color=INK, loc="left")
            style(ax)
            if i == 1:
                ax.set_xlabel("predicted error at the missing windows, 1 − mean max posterior (%)",
                              color=INK2, fontsize=8)
            if j == 0:
                ax.set_ylabel("observed MPM error at the missing windows (%)", color=INK2, fontsize=8)
    h, lab = axes[1, 1].get_legend_handles_labels()
    fig_.legend(h, lab, loc="lower center", ncol=5, frameon=False, fontsize=7)
    fig_.suptitle(f"Calibration at the missing windows (K = 2 and 3; zeros drawn at {ERR_FLOOR} %; "
                  "dashed: calibrated)", fontsize=10, color=INK, x=0.01, ha="left")
    fig_.tight_layout(rect=(0, 0.07, 1, 1))
    fig_.savefig(fig / "calibration_missing.png", dpi=130)
    plt.close(fig_)


def fig_errors(res: Path, fig: Path) -> None:
    c = _best_start(load_cls(res))
    c = c[c.model == "hmc_in"]
    cells = [("real", "ge10", None), ("real", "any", None), ("controlled", "ge10", "state"),
             ("controlled", "ge10", "state-markov"), ("controlled", "any", "state"),
             ("controlled", "any", "state-markov")]
    labels = ["real gaps\nge10", "real gaps\nany", "sim. 'state'\nge10 rates",
              "sim. Markov\nge10 rates", "sim. 'state'\nany rates", "sim. Markov\nany rates"]
    fig_, axes = plt.subplots(2, 2, figsize=(12.5, 6.6), sharex=True)
    for i, K in enumerate(pc.KS):
        for j, fit in enumerate(("oracle", "unsupervised")):
            ax = axes[i, j]
            variants = [v for v in VARIANTS if v in set(c[c.fit == fit].variant)]
            wbar = 0.8 / len(variants)
            for k, v in enumerate(variants):
                vals, errs = [], []
                for setting, rule, mask in cells:
                    x = c[(c.setting == setting) & (c.rule == rule) & (c.K == K) & (c.fit == fit)
                          & (c.variant == v)]
                    if mask is not None:
                        x = x[x["mask"] == mask]
                    e = 100 * x.err_missing.to_numpy()
                    vals.append(e.mean() if e.size else np.nan)
                    errs.append(e.std(ddof=1) / np.sqrt(e.size) if e.size > 1 else 0)
                pos = np.arange(len(cells)) + (k - (len(variants) - 1) / 2) * wbar
                ax.bar(pos, vals, wbar * 0.92, yerr=errs, color=VCOLOR[v], label=v,
                       error_kw={"lw": 0.8, "ecolor": INK2})
            ax.set_title(f"K = {K}, {fit} HMC-IN", fontsize=10, color=INK, loc="left")
            ax.set_xticks(np.arange(len(cells)), labels, fontsize=8)
            if j == 0:
                ax.set_ylabel("MPM error at the missing windows (%)", color=INK2, fontsize=9)
            style(ax)
            if i == 0:
                ax.legend(frameon=False, fontsize=7)
    fig_.tight_layout()
    fig_.savefig(fig / "errors_missing.png", dpi=130)
    plt.close(fig_)


def part_impute(res: Path) -> None:
    path = res / "imputation.csv"
    if not path.exists():
        return
    im = pd.read_csv(path)
    if "start_from" not in im:
        im["start_from"] = np.nan
    im["variant"] = np.where(im.start_from == "ignorable fit", im.mechanism + " (warm)", im.mechanism)
    emit("## 4. Imputation of the simulated gaps (truth known)\n")
    emit("Mean over subjects × mask seeds, all missing windows; posterior mean (RMSE), 200 FFBS "
         "draws (CRPS), 5–95 % posterior interval.\n")
    rows = []
    keys = ["rule", "mask", "K", "fit", "model", "variant"]
    a = im[im.subset == "all"]
    for key, x in a.groupby(keys, sort=False):
        r = dict(zip(keys, key))
        r.update({"M": x.n.mean(), "RMSE": x.rmse.mean(), "CRPS": x.crps.mean(),
                  "cov90": x.cov90.mean(), "width90": x.width90.mean()})
        rows.append(r)
    t = _sort(pd.DataFrame(rows), keys)
    fmt = {"M": "{:.0f}", "RMSE": "{:.4f}", "CRPS": "{:.4f}", "cov90": "{:.3f}", "width90": "{:.3f}"}
    emit(md(t, fmt))
    tex("imputation", t, fmt, caption="Imputation of the simulated gaps")
    emit()
    emit("By true group (K = 3, 'any' rates, Markov masks):\n")
    b = im[(im.K == 3) & (im.rule == "any") & (im["mask"] == "state-markov") & (im.subset != "all")]
    rows = []
    for key, x in b.groupby(["fit", "model", "subset", "variant"], sort=False):
        rows.append({"fit": key[0], "model": key[1], "group": key[2], "variant": key[3],
                     "n": x.n.mean(), "RMSE": x.rmse.mean(), "CRPS": x.crps.mean(),
                     "cov90": x.cov90.mean()})
    t = _sort(pd.DataFrame(rows), ["fit", "model", "group", "variant"])
    emit(md(t, {"n": "{:.0f}", "RMSE": "{:.4f}", "CRPS": "{:.4f}", "cov90": "{:.3f}"}))
    emit()


def part_runtime(res: Path) -> None:
    t = pd.read_csv(res / "tasks.csv")
    emit("## Runtime\n")
    g = t.groupby("task").agg(n=("seconds", "size"), wall_s=("seconds", "sum"),
                              cpu_s=("cpu_seconds", "sum"))
    emit(md(g.reset_index(), {"n": "{:.0f}", "wall_s": "{:.0f}", "cpu_s": "{:.0f}"}))
    emit(f"\nTotal: {t.cpu_seconds.sum() / 60:.1f} CPU-minutes, {t.seconds.sum() / 60:.1f} "
         f"task wall-minutes.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--figures", type=Path, default=HERE / "figures")
    ap.add_argument("--tables", type=Path, default=HERE / "tables")
    args = ap.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    args.tables.mkdir(parents=True, exist_ok=True)
    part_describe(args.results, args.figures)
    part_supervised(args.results)
    if (args.results / "lr_tests.csv").exists():
        part_tests(args.results)
    part_real(args.results)
    part_controlled(args.results, args.figures)
    part_basins(args.results)
    part_impute(args.results)
    part_runtime(args.results)
    (args.results / "tables.md").write_text("\n".join(OUT) + "\n")
    for name, body in TEX.items():
        (args.tables / f"{name}.tex").write_text(body)


if __name__ == "__main__":
    main()
