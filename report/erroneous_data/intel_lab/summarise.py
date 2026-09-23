"""
summarise.py — tables of the Intel Lab study (erroneous data), from the CSVs
of fit_clean.py, detect.py and robust.py: results/tables.md (every table) and
results/tables.tex (the key tables, booktabs).

Usage (from the repository root)
--------------------------------
    .venv/bin/python report/erroneous_data/intel_lab/summarise.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import il_common as C

R = C.RESULTS


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(out)


def esc(s) -> str:
    """LaTeX-escape a table cell."""
    return (str(s).replace("_", r"\_").replace("%", r"\%").replace("α", r"$\alpha$")
            .replace("±", r"$\pm$").replace("≥", r"$\geq$").replace("τ", r"$\tau$")
            .replace("π", r"$\pi$").replace("°C", r"\textdegree C").replace("–", "--")
            .replace("—", "---").replace("²", r"$^2$").replace("<", r"$<$"))


def tex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = list(df.columns)
    lines = [r"\begin{table}[ht]", r"\centering\small", rf"\caption{{{esc(caption)}}}",
             rf"\label{{{label}}}", r"\begin{tabular}{" + "l" * len(cols) + "}", r"\toprule",
             " & ".join(esc(c) for c in cols) + r" \\", r"\midrule"]
    for r in df.itertuples(index=False):
        lines.append(" & ".join(esc(v) for v in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def f2(x, n=2):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{n}f}"


def pval(p):
    if not np.isfinite(p):
        return "–"
    return "< 1e-300" if p == 0 else (f"{p:.2g}" if p < 0.01 else f"{p:.2f}")


def fits_table() -> pd.DataFrame:
    df = pd.read_csv(R / "fits.csv")
    rows = []
    for (m, k), g in df[df.best].groupby(["mote", "kind"], sort=False):
        b = {int(r.K): r for r in g.itertuples()}
        s = next(r for r in g.itertuples() if r.selected)
        rows.append({"mote": m, "model": C.KIND_LABEL[k],
                     "log-lik K=2": f"{b[2].loglik:.0f}", "log-lik K=3": f"{b[3].loglik:.0f}",
                     "BIC K=2": f"{b[2].bic:.0f}", "BIC K=3": f"{b[3].bic:.0f}",
                     "K": int(s.K), "means (°C)": s.mean, "sd (°C)": s.sd, "stay": s.stay,
                     "diagonal copulas (τ)": s.copulas if isinstance(s.copulas, str) else "–",
                     "start spread (nats)": f"{g[g.K == s.K].loglik.max() - df[(df.mote == m) & (df.kind == k) & (df.K == s.K)].loglik.min():.0f}"})
    return pd.DataFrame(rows)


def pit_table(part: str) -> pd.DataFrame:
    pc = pd.read_csv(R / "pit_checks.csv")
    pc.columns = [c.replace(".", "_").replace("-", "_") for c in pc.columns]
    pc = pc[(pc.part == part) & pc.selected]
    rows = []
    for r in pc.itertuples():
        rows.append({"mote": r.mote, "model": f"{C.KIND_LABEL[r.kind]} K={r.K}", "n": r.n,
                     "KS D (p)": f"{r.ks_stat:.3f} ({pval(r.ks_p)})",
                     "LB(10) z (p)": f"{r.lb_stat:.0f} ({pval(r.lb_p)})",
                     "LB(10) z² (p)": f"{r.lb2_stat:.0f} ({pval(r.lb2_p)})",
                     "z mean": f2(r.z_mean), "z sd": f2(r.z_sd), "acf1(z)": f2(r.acf1_z),
                     "flags α=1e-2 / 1e-3 / 1e-4 (expected)":
                         f"{getattr(r, 'nonseq_0_01')} / {getattr(r, 'nonseq_0_001')} / "
                         f"{getattr(r, 'nonseq_0_0001')} ({r.n * 1e-2:.0f} / {r.n * 1e-3:.0f} / {r.n * 1e-4:.1f})",
                     "seq flags α=1e-3": getattr(r, "seq_0_001")})
    return pd.DataFrame(rows)


def daynight_table() -> pd.DataFrame:
    """Share of the MPM state at night (00–06 h) and by day (10–16 h), days 1–10."""
    df = pd.read_csv(R / "regimes_by_hour.csv")
    part = df.hour.map(lambda h: "night" if h < 6 else ("day" if 10 <= h < 16 else ""))
    g = df[part != ""].assign(part=part[part != ""]).groupby(
        ["mote", "kind", "K", "state", "part"], sort=False).share.mean().unstack()
    rows = []
    for (m, k, K), gg in g.groupby(level=[0, 1, 2], sort=False):
        rows.append({"mote": m, "model": f"{C.KIND_LABEL[k]} K={K}",
                     "night share by state": " / ".join(f"{v:.2f}" for v in gg["night"]),
                     "day share by state": " / ".join(f"{v:.2f}" for v in gg["day"]),
                     "largest night − day gap": f"{(gg['night'] - gg['day']).abs().max():.2f}"})
    return pd.DataFrame(rows)


def detect_table(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    rows = []
    for meth in methods:
        for m in C.MOTES:
            r = df[(df.method == meth) & (df.mote == m)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append({"method": meth, "mote": m, "flags": int(r.n_flag),
                         "recall": f2(r.recall, 3), "recall rise": f2(r.recall_rise, 3),
                         "precision": f2(r.precision, 3), "precision ext": f2(r.precision_ext, 3),
                         "climb flagged": f"{int(r.flag_climb)}/{int(r.n_climb)}",
                         "ambig flagged": f"{int(r.flag_ambig)}/{int(r.n_ambig)}",
                         "normal flags ‰": f2(r.fa_per_1000, 2),
                         "FA episodes": int(r.fa_episodes),
                         "lead first (h)": f2(r.lead_first_h, 1),
                         "lead episode (h)": f2(r.lead_episode_h, 1)})
    return pd.DataFrame(rows)


def robust_table(kind: str) -> pd.DataFrame:
    df = pd.read_csv(R / "robust_fits.csv")
    df = df[df.kind == kind]
    rows = []
    for r in df.itertuples():
        pis = [float(x) for x in str(r.pi).split(" / ")]
        empty = sum(p < C.rc.DEGENERATE_MIN_PI for p in pis)
        rows.append({"mote": r.mote, "method": r.method, "fits": int(r.n_fits),
                     "converged": "–" if not isinstance(r.converged, (bool, np.bool_)) and pd.isna(r.converged)
                     else ("yes" if r.converged in (True, "True") else "no"),
                     "masked": int(r.n_masked),
                     "empty states": empty,
                     "mask: suspect / climb / normal": f"{r.mask_recall_suspect:.2f} / {r.mask_recall_climb:.2f} / {int(r.mask_normal)}",
                     "means (°C)": r.mean, "sd (°C)": r.sd, "π": r.pi,
                     "failure state (mean, sd, stay)": (f"{r.fail_mean:.1f}, {r.fail_sd:.1f}, {r.fail_stay:.4f}"
                                                        if r.fail_state else "none"),
                     "agree oracle_ext": f2(r.agree_oracle_ext, 3),
                     "agree clean model": f2(r.agree_clean_model, 3)})
    return pd.DataFrame(rows)


def robust_composition() -> pd.DataFrame:
    df = pd.read_csv(R / "robust_fits.csv")
    g = df.groupby("mote", sort=False).first().reset_index()
    rows = []
    for r in g.itertuples():
        rows.append({"mote": r.mote, "epochs": f"{r.lo}–{r.hi}", "observed": int(r.n_obs),
                     "normal": int(r.n_normal), "ambiguous 24 h": int(r.n_ambig),
                     "climb": int(r.n_climb), "suspect": int(r.n_suspect),
                     "suspect + climb (%)": f"{100 * (r.n_suspect + r.n_climb) / r.n_obs:.1f}"})
    return pd.DataFrame(rows)


def main():
    parts = ["# Intel Lab study — tables (generated by summarise.py)", ""]
    tex = []
    info = {}
    for f in ("fit_clean_info.json", "detect_info.json", "robust_info.json"):
        if (R / f).exists():
            info[f] = json.loads((R / f).read_text())
    parts += ["Run times (wall): " + ", ".join(
        f"{k.split('_info')[0]} {v.get('fit_seconds', 0) + v.get('pit_seconds', 0) + v.get('flag_seconds_wall', 0) + v.get('wall_seconds', 0):.0f} s"
        f" ({v.get('jobs')} processes)"
        for k, v in info.items()), ""]
    if "detect_info.json" in info:
        lab = info["detect_info.json"]["labels"]
        t = pd.DataFrame([{"mote": m, "first reading > 60 °C": f"{v['first_suspect']} ({str(C.epoch_time([v['first_suspect']])[0])[:16]})",
                           "climb onset": f"{v['climb_onset']} ({str(C.epoch_time([v['climb_onset']])[0])[:16]})",
                           "climb (h)": f"{(v['first_suspect'] - v['climb_onset']) / C.EPH:.1f}"}
                          for m, v in lab.items()])
        parts += ["## Failure dates", "", md_table(t), ""]
    t = fits_table()
    parts += ["## 1. Clean-window fits (days 1–7, best of 3 starts)", "", md_table(t), ""]
    tex.append(tex_table(t[["mote", "model", "BIC K=2", "BIC K=3", "K", "means (°C)", "sd (°C)",
                            "diagonal copulas (τ)"]],
                         "Clean-window fits (days 1--7): BIC and the selected model.", "tab:il-fits"))
    for part in ("held-out", "fit"):
        t = pit_table(part)
        parts += [f"## 1. PIT checks, {part} rows (selected K)", "", md_table(t), ""]
        if part == "held-out":
            tex.append(tex_table(t, "PIT checks on the held-out days 8--10.", "tab:il-pit"))
    t = daynight_table()
    parts += ["## 1. Regimes: MPM state share at night (00–06 h) and by day (10–16 h), days 1–10",
              "", "States sorted by margin mean.", "", md_table(t), ""]
    if (R / "detect_scores.csv").exists():
        df = pd.read_csv(R / "detect_scores.csv")
        main_m = ["PMC seq α=0.001", "PMC nonseq α=0.001", "HMC-IN seq α=0.001",
                  "HMC-IN nonseq α=0.001", "threshold 40 °C", "clean envelope ±2 °C",
                  "Hampel t=4", "robust z 24 h t=4"]
        t = detect_table(df, main_m)
        parts += ["## 2. Detection, main settings (epochs 28 801–102 706)", "", md_table(t), ""]
        tex.append(tex_table(t[["method", "mote", "flags", "recall", "precision", "precision ext",
                                "climb flagged", "normal flags ‰", "FA episodes",
                                "lead episode (h)"]],
                             "Detection on the later window: models at $\\alpha$ = 1e-3 and baselines.",
                             "tab:il-detect"))
        t = detect_table(df, list(dict.fromkeys(df.method)))
        parts += ["## 2. Detection, every setting", "", md_table(t), ""]
    if (R / "alarm_episodes.csv").exists():
        ep = pd.read_csv(R / "alarm_episodes.csv")
        t = ep.groupby(["method", "mote", "zone"]).agg(episodes=("n_flags", "size"),
                                                        flags=("n_flags", "sum")).reset_index()
        parts += ["## 2. Alarm episodes outside the suspect rows, by zone", "", md_table(t), ""]
        nz = ep[ep.zone == "normal"].copy()
        nz = nz[nz.method.isin(["PMC seq α=0.001", "PMC nonseq α=0.001"])]
        nz = nz.round({"hours": 2, "y_min": 2, "y_max": 2, "median_prev_hour": 2,
                       "jump_at_start": 2, "gap_before_h": 2})
        parts += ["## 2. PMC false-alarm episodes in normal operation (every one)", "",
                  md_table(nz[["method", "mote", "start", "hours", "n_flags", "y_min", "y_max",
                               "median_prev_hour", "jump_at_start", "gap_before_h"]]), ""]
    if (R / "robust_fits.csv").exists():
        t = robust_composition()
        parts += ["## 3. Mixed windows", "", md_table(t), ""]
        tex.append(tex_table(t, "Mixed windows of the robust estimation.", "tab:il-windows"))
        for kind in C.KINDS:
            t = robust_table(kind)
            parts += [f"## 3. Robust estimation, {C.KIND_LABEL[kind]}", "", md_table(t), ""]
            tex.append(tex_table(t[["mote", "method", "fits", "masked", "mask: suspect / climb / normal",
                                    "means (°C)", "failure state (mean, sd, stay)",
                                    "agree oracle_ext"]],
                                 f"Robust estimation on the mixed windows, {C.KIND_LABEL[kind]}.",
                                 f"tab:il-robust-{kind}"))
    (R / "tables.md").write_text("\n".join(parts))
    (R / "tables.tex").write_text("\n".join(tex))
    print("\n".join(parts))


if __name__ == "__main__":
    main()
