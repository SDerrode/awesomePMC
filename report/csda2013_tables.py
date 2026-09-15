"""Rebuild the LaTeX tables of ``csda2013_reproduction.tex`` from the committed CSV results.

No simulation is run. The script

* rewrites the Exp. 1/2 grid tables (``tables/exp[12]_*.tex`` and
  ``tables/margins_pair/exp[12]_*.tex``) in the orientation of CSDA
  Tables 2–5 (rows = tested copula, columns = true copula);
* writes the comparison tables of the report under ``tables/synthesis/``:
  Table 1 reading, Tables 2–5 against the paper, Exp. 3 against Tables 6–7,
  and the prior-width isolation of the Huard criterion.

Paper values: ``report/data/csda2013_tables2to5.csv`` (Tables 2–5 as
printed) and the constants below (Tables 1, 6, 7 and the error rates of
§4.3). Usage::

    python report/csda2013_tables.py
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

REPORT = Path(__file__).resolve().parent
RESULTS = REPORT / "results"
TABLES = REPORT / "tables"
SYNTH = TABLES / "synthesis"

_spec = importlib.util.spec_from_file_location("repro", REPORT / "reproduce_csda2013.py")
repro = importlib.util.module_from_spec(_spec)
sys.modules["repro"] = repro
_spec.loader.exec_module(repro)

# CSDA 2013 Table 1, Gamma margins G(λ, α, θ) as printed, with (μ, s) of the Gaussian margins.
TABLE1 = {
    "f_{11}": (0.0, 1.00, -2.83, 8.00, 0.13),
    "f_{12}": (0.3, 1.60, -3.65, 8.89, 0.18),
    "f_{21}": (1.1, 1.40, -2.08, 3.08, 0.46),
    "f_{22}": (1.5, 1.00, -1.63, 2.67, 0.38),
}
# CSDA 2013 Tables 6 (p_orig) and 7 (p_bal): right copula chosen out of 10 runs, pairs 11 12 21 22.
PAPER_EXP3 = {
    "exp1_orig_p": (10, 9, 7, 10), "exp2_orig_p": (10, 9, 9, 10),
    "exp1_bal_p": (10, 10, 10, 10), "exp2_bal_p": (10, 10, 10, 8),
}
# CSDA 2013 §4.3 mean error rates (p_orig only): (unsupervised, supervised).
PAPER_EXP3_ERR = {"exp1_orig_p": (13.74, 12.16), "exp2_orig_p": (12.89, 11.35)}
CFGS = ["exp1_orig_p", "exp2_orig_p", "exp1_bal_p", "exp2_bal_p"]
CFG_TEX = {
    "exp1_orig_p": r"exp.~1, $p_{\text{orig}}$", "exp2_orig_p": r"exp.~2, $p_{\text{orig}}$",
    "exp1_bal_p": r"exp.~1, $p_{\text{bal}}$", "exp2_bal_p": r"exp.~2, $p_{\text{bal}}$",
}
GRIDS = ["low_tau_gauss", "low_tau_gamma", "high_tau_gauss", "high_tau_gamma"]
PAPER_NAME = {("exp1", "low_tau_gauss"): "2a", ("exp1", "low_tau_gamma"): "2b",
              ("exp1", "high_tau_gauss"): "3a", ("exp1", "high_tau_gamma"): "3b",
              ("exp2", "low_tau_gauss"): "4a", ("exp2", "low_tau_gamma"): "4b",
              ("exp2", "high_tau_gauss"): "5a", ("exp2", "high_tau_gamma"): "5b"}


def read_grid(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """CSV grid → (copulas, means[sim, est], stds[sim, est])."""
    lines = [ln for ln in path.read_text().splitlines() if not ln.startswith('"#')]
    cops = lines[0].split(",")[1:]
    cells = [[c.strip() for c in ln.split(",")[1:]] for ln in lines[1:]]
    means = np.array([[float(c.split(" ")[0]) for c in row] for row in cells])
    stds = np.array([[float(c.split("(")[1].rstrip(")")) for c in row] for row in cells])
    return cops, means, stds


def paper_grid(name: str, cops: list[str]) -> np.ndarray:
    """Paper table → means[sim(true), est(tested)], same indexing as the CSVs."""
    df = pd.read_csv(REPORT / "data" / "csda2013_tables2to5.csv", skiprows=1)
    df = df[df.table == name]
    out = np.full((len(cops), len(cops)), np.nan)
    for _, r in df.iterrows():
        out[cops.index(r.true), cops.index(r.tested)] = r["mean"]
    return out


def rebuild_grids() -> None:
    for key, d in (("state", RESULTS), ("pair", RESULTS / "margins_pair")):
        res = {}
        for exp in ("exp1", "exp2"):
            res[exp] = {}
            for g in GRIDS:
                cops, m, s = read_grid(d / f"{exp}_{g}.csv")
                res[exp][g] = {"copulas": cops, "means": m, "stds": s}
        repro.write_all_latex_tables(res["exp1"], res["exp2"], None, margins=key)


def exp3_runs(d: Path, cfg: str, crit: str) -> pd.DataFrame:
    return pd.read_csv(d / f"exp3_{cfg}__{crit}__runs.csv")


def exp3_err(d: Path, cfg: str, crit: str) -> dict[str, float]:
    out = {}
    for line in (d / f"exp3_{cfg}__{crit}.csv").read_text().splitlines():
        p = line.split(",")
        if len(p) == 2 and "err_mean" in p[0]:
            out[p[0].split("_err")[0]] = float(p[1])
    return out


def mcnemar(a: pd.DataFrame, b: pd.DataFrame, keys=("run", "i", "j")) -> tuple[int, int, float]:
    m = pd.merge(a, b, on=list(keys), suffixes=("_a", "_b"))
    n10 = int(((m.hit_a == 1) & (m.hit_b == 0)).sum())
    n01 = int(((m.hit_a == 0) & (m.hit_b == 1)).sum())
    p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    return n10, n01, float(p)


def tex_p(p: float) -> str:
    if p >= 0.01:
        return f"{p:.2f}"
    mant, exp = f"{p:.1e}".split("e")
    return rf"{mant}\times10^{{{int(exp)}}}"


def write(name: str, lines: list[str]) -> None:
    SYNTH.mkdir(parents=True, exist_ok=True)
    (SYNTH / name).write_text("\n".join(lines) + "\n")


def table1_reading() -> None:
    lines = [r"\begin{tabular}{lcc@{\hspace{1.4em}}cc@{\hspace{1.4em}}cc@{\hspace{1.4em}}cc}", r"\toprule",
             r" & \multicolumn{2}{c}{Gaussian $\mathcal N(\mu,s)$} & \multicolumn{2}{c}{printed $\mathcal G$}"
             r" & \multicolumn{2}{c}{$s$ = std} & \multicolumn{2}{c}{$s$ = variance} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
             r"margin & $\mu$ & $s$ & $\lambda$ & $\theta$ & $\lambda$ & $\theta$ & $\lambda$ & $\theta$ \\", r"\midrule"]
    for name, (mu, s, lam, a, th) in TABLE1.items():
        sd_std, sd_var = s, math.sqrt(s)
        lines.append(rf"${name}$ & {mu:.1f} & {s:.2f} & {lam:.2f} & {th:.2f} & "
                     rf"{-math.sqrt(a) * sd_std:.2f} & {(sd_std / math.sqrt(a)) ** 2:.3f} & "
                     rf"{-math.sqrt(a) * sd_var:.2f} & {(sd_var / math.sqrt(a)) ** 2:.3f} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("table1_reading.tex", lines)


def tables2to5_vs_paper() -> None:
    lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
             r" & & \multicolumn{2}{c}{mean $|\Delta|$ (points)} & \multicolumn{2}{c}{mean $\Delta$ on the diagonal}"
             r" & \multicolumn{2}{c}{correlation} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
             r"paper table & setting & state & pair & state & pair & state & pair \\", r"\midrule"]
    labels = {"low_tau_gauss": r"$\tau=0.16$, Gauss", "low_tau_gamma": r"$\tau=0.16$, Gamma",
              "high_tau_gauss": r"$\tau=0.70$, Gauss", "high_tau_gamma": r"$\tau=0.70$, Gamma"}
    for exp in ("exp1", "exp2"):
        for g in GRIDS:
            name = PAPER_NAME[(exp, g)]
            stats = []
            for d in (RESULTS, RESULTS / "margins_pair"):
                cops, m, _ = read_grid(d / f"{exp}_{g}.csv")
                pg = paper_grid(name, cops)
                diff = m - pg
                stats.append((np.abs(diff).mean(), np.diag(diff).mean(), np.corrcoef(m.ravel(), pg.ravel())[0, 1]))
            (a1, d1, c1), (a2, d2, c2) = stats
            model = "PMC" if exp == "exp1" else "PMM"
            lines.append(rf"{name[0]}({name[1]}) & {model}, {labels[g]} & {a1:.2f} & \textbf{{{a2:.2f}}} & "
                         rf"${d1:+.2f}$ & ${d2:+.2f}$ & {c1:.3f} & {c2:.3f} \\")
        if exp == "exp1":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tables2to5_vs_paper.tex", lines)


def exp3_vs_paper(crit: str = "huard") -> None:
    single, multi = RESULTS / "margins_pair", RESULTS / "exp3_multistart" / "margins_pair"
    lines = [r"\begin{tabular}{llccccccc}", r"\toprule",
             r" & & \multicolumn{4}{c}{right family chosen (\%)} & \multicolumn{2}{c}{error (\%)} \\",
             r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}",
             r"setting & source & $c_{11}$ & $c_{12}$ & $c_{21}$ & $c_{22}$ & unsup. & sup. \\", r"\midrule"]
    for cfg in CFGS:
        paper = PAPER_EXP3[cfg]
        pe = PAPER_EXP3_ERR.get(cfg)
        lines.append(rf"{CFG_TEX[cfg]} & paper (10 runs) & " + " & ".join(f"{10 * h:.0f}" for h in paper)
                     + (rf" & {pe[0]:.2f} & {pe[1]:.2f} \\" if pe else r" & -- & -- \\"))
        for d, lab in ((single, "single start"), (multi, "multistart")):
            r = exp3_runs(d, cfg, crit)
            n = r.run.nunique()
            hits = [100 * r[(r.i == i) & (r.j == j)].hit.mean() for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))]
            e = exp3_err(d, cfg, crit)
            lines.append(rf" & {lab} ({n} runs) & " + " & ".join(f"{h:.0f}" for h in hits)
                         + rf" & {e['unsup']:.1f} & {e['sup']:.1f} \\")
        if cfg != CFGS[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(f"exp3_vs_paper_{crit}.tex", lines)


def huard_cells_state() -> None:
    lines = [r"\begin{tabular}{llccc}", r"\toprule",
             r"setting & true family (width) & Huard ($c_{12}$, $c_{21}$) & MLE ($c_{12}$, $c_{21}$) & discordance \\",
             r"\midrule"]
    for cfg in CFGS:
        h, m = exp3_runs(RESULTS, cfg, "huard"), exp3_runs(RESULTS, cfg, "mle")
        hh, mm, dd = [], [], []
        for i, j in ((0, 1), (1, 0)):
            a = h[(h.i == i) & (h.j == j)]
            b = m[(m.i == i) & (m.j == j)]
            hh.append(f"{100 * a.hit.mean():.0f}\\%")
            mm.append(f"{100 * b.hit.mean():.0f}\\%")
            n10, n01, _ = mcnemar(a, b)
            dd.append(f"{n10}:{n01}")
        fam = h[(h.i == 0) & (h.j == 1)].true_name.iloc[0]
        width = "1.000" if fam == "GH" else "0.165"
        lines.append(rf"{CFG_TEX[cfg]} & {fam} ({width}) & {', '.join(hh)} & {', '.join(mm)} & {', '.join(dd)} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("huard_offdiag_cells_state.tex", lines)


def prior_width_state() -> None:
    pw = pd.read_csv(RESULTS / "prior_width_isolation.csv")
    pw = pw[pw.i != pw.j]
    lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
             r"setting & true family & MLE & Huard & common support & edge retained & Huard vs common & $p$ \\",
             r"\midrule"]
    for cfg in CFGS:
        s = pw[pw.config == cfg]
        rate = {c: 100 * s[s.criterion == c].hit.mean() for c in ("mle", "huard", "huard_common")}
        edge = 100 * (rate["huard_common"] - rate["mle"]) / (rate["huard"] - rate["mle"])
        n10, n01, p = mcnemar(s[s.criterion == "huard"], s[s.criterion == "huard_common"])
        fam = s.true.iloc[0]
        lines.append(rf"{CFG_TEX[cfg]} & {fam} & {rate['mle']:.1f}\% & {rate['huard']:.1f}\% & "
                     rf"{rate['huard_common']:.1f}\% & {edge:.0f}\% & {n10}:{n01} & ${tex_p(p)}$ \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("prior_width_isolation_state.tex", lines)


def multistart_criterion_summary() -> None:
    src = TABLES / "exp3_multistart" / "margins_pair" / "exp4_criterion_summary.tex"
    text = src.read_text().replace(r"\label{tab:exp4-summary}", r"\label{tab:exp4-multi}")
    text = text.replace(r"\caption{Selection-criterion comparison (summary).",
                        r"\caption{Selection-criterion comparison, pair margins and multistart ICE (30 runs).")
    write("exp4_criterion_summary_multistart.tex", text.rstrip("\n").splitlines())


def main() -> None:
    rebuild_grids()
    multistart_criterion_summary()
    table1_reading()
    tables2to5_vs_paper()
    for crit in ("huard", "mle"):
        exp3_vs_paper(crit)
    huard_cells_state()
    prior_width_state()
    print("tables written under", TABLES)


if __name__ == "__main__":
    main()
