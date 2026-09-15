"""
summarise.py — LaTeX tables and PNG figures of the missing-data benchmark.

Reads ``runs.csv`` written by ``run_benchmark.py`` (no simulation) and writes

``report/tables/missing_benchmark/``
    ``headline.tex``            error on missing positions at 20 % and 40 %,
                                complete / marginalise / plugin / linear;
    ``err_missing_<model>.tex`` error on missing positions, every pattern,
                                method and rate, mean (sd) over the reps;
    ``imputation_<model>.tex``  RMSE and CRPS of the imputations;
    ``coverage90.tex``, ``coverage95.tex``
                                interval coverage, marginalise vs plugin;
    ``overconfidence.tex``      posterior confidence vs accuracy on the missing
                                positions at 40 %;
    ``timing.tex``              median wall time per sequence and method.

``report/figures/missing_benchmark/``
    ``error_missing_<pattern>.png``, ``crps_<pattern>.png``,
    ``coverage90_<pattern>.png`` — one panel per model, one line per method,
    the missing rate on the x axis.

Usage (from the repository root)::

    .venv/bin/python report/missing_benchmark/summarise.py
    .venv/bin/python report/missing_benchmark/summarise.py \\
        --results report/results/missing_benchmark/quick \\
        --tables /tmp/mb_tables --figures /tmp/mb_figures
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from run_benchmark import (  # noqa: E402
    METHODS, MODELS, PATTERNS, REPORT_DIR, RESULTS_DIR, read_runs, summarise_runs,
)

TABLES_DIR = REPORT_DIR / "tables" / "missing_benchmark"
FIGURES_DIR = REPORT_DIR / "figures" / "missing_benchmark"

#: Fixed categorical slots (validated palette, light mode) + a marker per method
#: as secondary encoding; the complete-data bound is a neutral dashed reference.
METHOD_STYLE: dict[str, dict] = {
    "marginalise": {"color": "#2a78d6", "marker": "o", "label": "marginalise (exact)"},
    "plugin": {"color": "#eb6834", "marker": "s", "label": "plug-in posterior mean"},
    "linear": {"color": "#1baf7a", "marker": "^", "label": "linear interpolation"},
    "locf": {"color": "#eda100", "marker": "D", "label": "LOCF"},
    "mean": {"color": "#e87ba4", "marker": "v", "label": "observed mean"},
    "complete": {"color": "#898781", "marker": None, "linestyle": "--",
                 "label": "complete data (lower bound)"},
}
MODEL_SHORT: dict[str, str] = {
    "hmc_in_gauss_k2": "HMC-IN Gauss",
    "hmc_dn_gauss_k2": "HMC-DN Gauss",
    "pmc_gauss_k2": "PMC state Gauss",
    "pmc_pair_gauss_k2": "PMC pair Gauss--Clayton",
    "pmc_pair_gamma_gumbel_k2": "PMC pair Gamma--Gumbel",
}
PATTERN_SHORT: dict[str, str] = {
    "mcar_b1": "MCAR $b{=}1$", "mcar_b10": "MCAR $b{=}10$", "mcar_b50": "MCAR $b{=}50$",
    "scattered": "scattered", "aligned": "aligned",
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

class Summary:
    """Summary rows indexed by (model, pattern, rate, method)."""

    def __init__(self, runs: list[dict]):
        self.runs = runs
        self.rows = {(s["model"], s["pattern"], s["rate"], s["method"]): s
                     for s in summarise_runs(runs)}
        self.models = [m for m in MODELS if any(k[0] == m for k in self.rows)]
        self.patterns = [p for p in PATTERNS if any(k[1] == p for k in self.rows)]
        self.rates = sorted({k[2] for k in self.rows})
        self.methods = [m for m in METHODS if any(k[3] == m for k in self.rows)]
        self.reps = max(s["n_reps"] for s in self.rows.values())
        self.n_obs = int(runs[0]["n"])

    def get(self, model, pattern, rate, method, col, stat="mean") -> float:
        s = self.rows.get((model, pattern, rate, method))
        return math.nan if s is None else float(s[f"{col}_{stat}"])


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------

def _num(v: float, scale: float = 1.0, digits: int = 1) -> str:
    return "--" if not math.isfinite(v) else f"{scale * v:.{digits}f}"


def _pct(r: float) -> str:
    return f"{100 * r:g}\\,\\%"


def _table(path: Path, caption: str, label: str, colspec: str, header: list[str],
           body: list[str]) -> None:
    lines = [
        r"\begin{table}[htbp]", r"  \centering", r"  \small", r"  \setlength{\tabcolsep}{4pt}",
        f"  \\caption{{{caption}}}", f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{colspec}}}", r"    \toprule",
        *[f"    {h}" for h in header], r"    \midrule",
        *[f"    {b}" for b in body], r"    \bottomrule", r"  \end{tabular}", r"\end{table}", "",
    ]
    path.write_text("\n".join(lines))


def write_tables(S: Summary, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    note = f"Mean over {S.reps} replicates, $N = {S.n_obs}$, known model."

    # headline: error on missing positions at two rates
    head_rates = [r for r in (0.2, 0.4) if r in S.rates] or S.rates[-2:]
    head_methods = [m for m in ("complete", "marginalise", "plugin", "linear") if m in S.methods]
    body = []
    for model in S.models:
        for k, pat in enumerate(S.patterns):
            first = MODEL_SHORT[model] if k == 0 else ""
            cells = [_num(S.get(model, pat, r, m, "err_missing"), 100)
                     for r in head_rates for m in head_methods]
            body.append(f"{first} & {PATTERN_SHORT[pat]} & " + " & ".join(cells) + r" \\")
        body.append(r"\addlinespace")
    body = body[:-1]
    ncol = len(head_methods)
    header = [
        " & & " + " & ".join(f"\\multicolumn{{{ncol}}}{{c}}{{missing rate {_pct(r)}}}"
                             for r in head_rates) + r" \\",
        "".join(f"\\cmidrule(lr){{{3 + i * ncol}-{2 + (i + 1) * ncol}}}"
                for i in range(len(head_rates))),
        "model & pattern & " + " & ".join(m for _ in head_rates for m in head_methods) + r" \\",
    ]
    p = out / "headline.tex"
    _table(p, "Classification error (\\%) on the missing positions. " + note,
           "tab:missing-headline", "ll" + "r" * (ncol * len(head_rates)), header, body)
    written.append(p)

    # full error grid per model
    for model in S.models:
        body = []
        for pat in S.patterns:
            for k, m in enumerate(S.methods):
                first = PATTERN_SHORT[pat] if k == 0 else ""
                cells = [f"{_num(S.get(model, pat, r, m, 'err_missing'), 100)} "
                         f"({_num(S.get(model, pat, r, m, 'err_missing', 'sd'), 100)})"
                         for r in S.rates]
                body.append(f"{first} & {m} & " + " & ".join(cells) + r" \\")
            body.append(r"\addlinespace")
        header = ["pattern & method & " + " & ".join(_pct(r) for r in S.rates) + r" \\"]
        p = out / f"err_missing_{model}.tex"
        _table(p, f"{MODELS[model][0]}: classification error (\\%) on the missing positions, "
               f"mean (sd). " + note, f"tab:missing-err-{model}",
               "ll" + "c" * len(S.rates), header, body[:-1])
        written.append(p)

    # imputation per model
    imp_methods = [m for m in S.methods if m != "complete"]
    for model in S.models:
        body = []
        for pat in S.patterns:
            for k, m in enumerate(imp_methods):
                first = PATTERN_SHORT[pat] if k == 0 else ""
                cells = [f"{_num(S.get(model, pat, r, m, 'rmse'), digits=2)} / "
                         f"{_num(S.get(model, pat, r, m, 'crps'), digits=3)}" for r in S.rates]
                body.append(f"{first} & {m} & " + " & ".join(cells) + r" \\")
            body.append(r"\addlinespace")
        header = ["pattern & method & " + " & ".join(_pct(r) for r in S.rates) + r" \\"]
        p = out / f"imputation_{model}.tex"
        _table(p, f"{MODELS[model][0]}: imputation RMSE / CRPS on the missing positions. "
               "CRPS from 200 posterior draws (marginalise), Gaussian with the posterior sd "
               "(plugin), Gaussian with the sd of the observed values (linear, locf, mean). "
               + note, f"tab:missing-imp-{model}", "ll" + "c" * len(S.rates), header, body[:-1])
        written.append(p)

    # coverage: marginalise quantiles vs plugin Gaussian band
    for level, z in ((90, "1.645"), (95, "1.96")):
        body = []
        for model in S.models:
            for k, pat in enumerate(S.patterns):
                first = MODEL_SHORT[model] if k == 0 else ""
                cells = [f"{_num(S.get(model, pat, r, 'marginalise', f'cov{level}'), 100)} / "
                         f"{_num(S.get(model, pat, r, 'plugin', f'cov{level}'), 100)}"
                         for r in S.rates]
                body.append(f"{first} & {PATTERN_SHORT[pat]} & " + " & ".join(cells) + r" \\")
            body.append(r"\addlinespace")
        header = ["model & pattern & " + " & ".join(_pct(r) for r in S.rates) + r" \\"]
        lo, hi = (5, 95) if level == 90 else (2.5, 97.5)
        p = out / f"coverage{level}.tex"
        _table(p, f"Coverage (\\%) of the nominal {level}\\,\\% intervals on the missing "
               f"positions: exact posterior quantiles $[q_{{{lo:g}}}, q_{{{hi:g}}}]$ "
               f"(marginalise) / posterior mean $\\pm {z}$ posterior sd (plugin). " + note,
               f"tab:missing-cov{level}", "ll" + "c" * len(S.rates), header, body[:-1])
        written.append(p)

    # overconfidence at 40 %
    rate = 0.4 if 0.4 in S.rates else S.rates[-1]
    oc_methods = [m for m in ("marginalise", "plugin", "linear", "mean") if m in S.methods]
    body = []
    for model in S.models:
        for k, pat in enumerate(S.patterns):
            first = MODEL_SHORT[model] if k == 0 else ""
            cells = []
            for m in oc_methods:
                cells += [_num(S.get(model, pat, rate, m, "conf_missing"), 100),
                          _num(1.0 - S.get(model, pat, rate, m, "err_missing"), 100),
                          _num(S.get(model, pat, rate, m, "logloss_missing"), digits=2)]
            body.append(f"{first} & {PATTERN_SHORT[pat]} & " + " & ".join(cells) + r" \\")
        body.append(r"\addlinespace")
    header = [
        " & & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{m}}}" for m in oc_methods) + r" \\",
        "".join(f"\\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(oc_methods))),
        "model & pattern & " + " & ".join("conf & acc & LL" for _ in oc_methods) + r" \\",
    ]
    p = out / "overconfidence.tex"
    _table(p, f"Missing positions at rate {_pct(rate)}: mean posterior confidence "
           "$\\max_k \\gamma_n(k)$ (\\%), accuracy (\\%) and log-loss $-\\log \\gamma_n(x_n)$. "
           "A calibrated classifier has conf $\\approx$ acc. " + note,
           "tab:missing-overconfidence", "ll" + "rrr" * len(oc_methods), header, body[:-1])
    written.append(p)

    # timing: median over every task of a model
    cols = [("complete", "time_total_s", "complete"),
            ("marginalise", "time_classify_s", "marg.\\ classify"),
            ("marginalise", "time_fill_s", "marg.\\ impute+draws"),
            ("plugin", "time_total_s", "plugin"),
            ("linear", "time_total_s", "linear"),
            ("locf", "time_total_s", "locf"),
            ("mean", "time_total_s", "mean")]
    cols = [c for c in cols if c[0] in S.methods]
    body = []
    for model in S.models:
        cells = []
        for method, col, _ in cols:
            v = [r[col] for r in S.runs if r["model"] == model and r["method"] == method]
            cells.append(_num(float(np.median(v)), 1000.0))
        body.append(f"{MODEL_SHORT[model]} & " + " & ".join(cells) + r" \\")
    header = ["model & " + " & ".join(c[2] for c in cols) + r" \\"]
    p = out / "timing.tex"
    _table(p, "Median wall time (ms) per sequence over every pattern, rate and replicate, "
           f"$N = {S.n_obs}$, one BLAS thread per worker (four workers running side by side). "
           "marg.\\ classify: \\texttt{classify} on the gapped series; "
           "marg.\\ impute+draws: \\texttt{impute} with five "
           "quantiles and 200 FFBS draws; plugin: \\texttt{impute} (mean and sd only) + "
           "\\texttt{classify}.", "tab:missing-timing", "l" + "r" * len(cols), header, body)
    written.append(p)
    return written


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _style_axis(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _panel_figure(S: Summary, pattern: str, col: str, methods: list[str], ylabel: str,
                  title: str, path: Path, *, scale: float = 1.0, reference: float | None = None,
                  reference_label: str = "") -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), sharex=True, sharey=True)
    axes = axes.ravel()
    x = 100 * np.array(S.rates)
    for ax, model in zip(axes, S.models):
        _style_axis(ax)
        if reference is not None:
            ax.axhline(reference, color=MUTED, linewidth=1.0, linestyle=":", zorder=1)
        for m in methods:
            st = METHOD_STYLE[m]
            y = scale * np.array([S.get(model, pattern, r, m, col) for r in S.rates])
            ax.plot(x, y, color=st["color"], marker=st["marker"], markersize=5.5,
                    linewidth=1.6, linestyle=st.get("linestyle", "-"), label=st["label"],
                    markeredgecolor="white", markeredgewidth=0.8,
                    zorder=4 if m == "marginalise" else 3)   # the exact method on top
        ax.set_title(MODELS[model][0], fontsize=10, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in x])
    for ax in axes[len(S.models):]:
        ax.axis("off")
    for i, ax in enumerate(axes[:len(S.models)]):
        if i + 3 >= len(S.models):          # no model panel below: label this x axis
            ax.set_xlabel("missing rate (%)", fontsize=10, color=MUTED)
            ax.tick_params(labelbottom=True)
    for ax in axes[::3]:
        ax.set_ylabel(ylabel, fontsize=10, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    if reference is not None:
        handles.append(plt.Line2D([], [], color=MUTED, linewidth=1.0, linestyle=":"))
        labels.append(reference_label)
    legend_ax = axes[len(S.models)] if len(S.models) < 6 else axes[-1]
    legend_ax.legend(handles, labels, loc="center", frameon=False, fontsize=10,
                     title=f"{PATTERNS[pattern]}\nmean over {S.reps} reps, N = {S.n_obs}",
                     title_fontsize=10)
    fig.suptitle(title, fontsize=12, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)


def write_figures(S: Summary, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    imp_methods = [m for m in METHODS if m != "complete" and m in S.methods]
    for pat in S.patterns:
        p = out / f"error_missing_{pat}.png"
        _panel_figure(S, pat, "err_missing", S.methods, "error on missing positions (%)",
                      "Classification error on the missing positions", p, scale=100.0)
        written.append(p)
        p = out / f"crps_{pat}.png"
        _panel_figure(S, pat, "crps", imp_methods, "CRPS (lower is better)",
                      "Imputation CRPS on the missing positions (marginalise: posterior draws; "
                      "plugin: Gaussian, posterior sd; others: Gaussian, sd of observed y)", p)
        written.append(p)
        p = out / f"coverage90_{pat}.png"
        _panel_figure(S, pat, "cov90", imp_methods, "coverage of the 90 % interval (%)",
                      "Coverage of the nominal 90 % intervals on the missing positions", p,
                      scale=100.0, reference=90.0, reference_label="nominal 90 %")
        written.append(p)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Tables and figures of the missing-data benchmark.")
    ap.add_argument("--results", type=Path, default=RESULTS_DIR,
                    help="directory holding runs.csv")
    ap.add_argument("--tables", type=Path, default=TABLES_DIR)
    ap.add_argument("--figures", type=Path, default=FIGURES_DIR)
    args = ap.parse_args(argv)
    S = Summary(read_runs(args.results / "runs.csv"))
    files = write_tables(S, args.tables) + write_figures(S, args.figures)
    print(f"{len(files)} files written to {args.tables} and {args.figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
