"""
summarise.py — tables and figures of the forecasting study, from the result CSVs
only (forecasting).

Prints every table of ``README.md`` (Markdown), writes them to
``<results>/tables.md`` and the main ones to ``<results>/tables.tex``; figures
go to ``--figures``.

Usage (repository root)::

    .venv/bin/python report/forecasting/summarise.py
    .venv/bin/python report/forecasting/summarise.py --results report/out/forecasting/quick/results \
        --figures report/out/forecasting/quick/figures
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

# Categorical slots (fixed order, validated palette of report/real_series) and inks.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dddcd6"

SERIES = {"aotizhongxin": "Beijing Aotizhongxin (ln PM2.5, 1 h)",
          "tsnh4": "tsNH4 (ln NH4, 10 min)",
          "mote20": "Intel Lab mote 20 (°C, 30 s)"}
FIXTURES = {"hmc_in_gauss_k2": "HMC-IN fixture", "pmc_gauss_k2": "PMC fixture"}
FAM_LABEL = {"pmc_pair_K3": "PMC pair K=3", "pmc_state_K2": "PMC state K=2", "pmc_state_K3": "PMC state K=3",
             "pmc_pair_K2": "PMC pair K=2", "hmc_in_K3": "HMC-IN K=3", "hmc_in_K2": "HMC-IN K=2",
             "pmm": "PMM (pmmforecast)", "pmm_default": "PMM, default starts", "ar1": "AR(1)",
             "persistence": "persistence"}
VARIANT_LABEL = {"raw": "raw", "raw+gate": "raw + gate", "robust": "robust", "robust+gate": "robust + gate",
                 "hampel+gate": "Hampel robust + gate", "clean": "clean fit", "true": "true model",
                 "true+gate": "true model + gate"}
# real_series §4 (report/real_series/results/forecast_beijing.csv) method names
S4_NAMES = {"pmc_pair_K3|raw": "pmc_pair_K3", "hmc_in_K3|raw": "hmc_in_K3", "ar1|raw": "ar1",
            "persistence|raw": "persistence"}

OUT: list[str] = []
TEX: list[str] = []


def emit(text: str = "") -> None:
    print(text)
    OUT.append(text)


def fmt_cell(v, f) -> str:
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "–"
        return f(v) if callable(f) else f.format(v)
    return str(v)


def md(df: pd.DataFrame, fmt: dict | None = None, default: str = "{:.3f}") -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt_cell(r[c], fmt.get(c, default)) for c in cols) + " |")
    return "\n".join(lines)


def tex(df: pd.DataFrame, caption: str, fmt: dict | None = None, default: str = "{:.3f}") -> None:
    fmt = fmt or {}
    cols = list(df.columns)

    def esc(s):
        return (str(s).replace("_", r"\_").replace("%", r"\%").replace("|", "/").replace("&", r"\&")
                .replace("τ", r"$\tau$").replace("≤", r"$\le$").replace("−", "-").replace("×", r"$\times$"))
    TEX.append(r"\begin{table}[ht]\centering\small")
    TEX.append(rf"\caption{{{esc(caption)}}}")
    TEX.append(r"\begin{tabular}{" + "l" * 1 + "r" * (len(cols) - 1) + "}")
    TEX.append(r"\hline")
    TEX.append(" & ".join(esc(c) for c in cols) + r" \\ \hline")
    for _, r in df.iterrows():
        TEX.append(" & ".join(esc(fmt_cell(r[c], fmt.get(c, default))) for c in cols) + r" \\")
    TEX.append(r"\hline\end{tabular}\end{table}")
    TEX.append("")


def section(title: str, level: int = 2) -> None:
    emit("")
    emit("#" * level + " " + title)
    emit("")


def mlabel(method: str) -> str:
    fam, var = method.split("|")
    return f"{FAM_LABEL.get(fam, fam)}" + ("" if var == "raw" else f", {VARIANT_LABEL.get(var, var)}")


def style(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def part1_methods(df: pd.DataFrame, study: str) -> list[str]:
    ms = df[df.study == study].method.unique().tolist()
    pmc = sorted([m for m in ms if m.startswith("pmc_") and m.endswith("|raw")])
    hmc = sorted([m for m in ms if m.startswith("hmc_in") and m.endswith("|raw")])
    rest = [m for m in ("pmm|raw", "pmm_default|raw", "ar1|raw", "persistence|raw") if m in ms]
    return pmc + hmc + rest


# ---------------------------------------------------------------------------
# Part 1
# ---------------------------------------------------------------------------

def part1(R: Path, F: Path) -> None:
    by_h = pd.read_csv(R / "part1_by_h.csv")
    allh = pd.read_csv(R / "part1_all_h.csv")
    section("Part 1 — forecasts on real series")
    for study in [s for s in SERIES if s in by_h.study.unique()]:
        meths = part1_methods(by_h, study)
        g = by_h[by_h.study == study]
        H = int(g.h.max())
        hs = [h for h in (1, 2, 3, 6, 12, 24) if h <= H]
        section(SERIES[study], 3)
        for metric, name in (("rmse", "RMSE"), ("crps", "CRPS")):
            t = g[g.h.isin(hs)].pivot(index="h", columns="method", values=metric)[meths]
            t.columns = [mlabel(m) for m in t.columns]
            t = t.reset_index()
            a = allh[allh.study == study].set_index("method")
            mean_row = {"h": "mean 1–%d" % H}
            for m in meths:
                mean_row[mlabel(m)] = float(np.sqrt(a.loc[m, "mse"])) if metric == "rmse" else float(a.loc[m, "crps"])
            t = pd.concat([t.astype({"h": str}), pd.DataFrame([mean_row])], ignore_index=True)
            emit(f"{name} by horizon (n = {int(g[g.h == 1].n.iloc[0])} origins at h = 1):")
            emit("")
            emit(md(t, {"h": "{}"}))
            emit("")
            tex(t, f"{SERIES[study]}: {name} by horizon", {"h": "{}"})
        rows = []
        for m in meths:
            a = allh[(allh.study == study) & (allh.method == m)].iloc[0]
            g1 = g[(g.method == m) & (g.h == 1)].iloc[0]
            gH = g[(g.method == m) & (g.h == H)].iloc[0]
            rows.append({"model": mlabel(m), "cov50": a.cov50, "cov80": a.cov80, "cov95": a.cov95,
                         "cov95 h=1": g1.cov95, f"cov95 h={H}": gH.cov95, "width95": a.w95,
                         "CRPS (Gaussian approx.)": a.crps_gauss, "CRPS": a.crps})
        emit("Coverage of the central 50 / 80 / 95 % intervals (all horizons), width of the 95 % "
             "interval, and CRPS against the Gaussian CRPS of the predictive mean and sd:")
        emit("")
        emit(md(pd.DataFrame(rows), {"cov50": "{:.2f}", "cov80": "{:.2f}", "cov95": "{:.2f}",
                                      "cov95 h=1": "{:.2f}", f"cov95 h={H}": "{:.2f}"}))
        emit("")

    # §4 reproduction
    s4p = HERE.parent / "real_series" / "results" / "forecast_beijing.csv"
    if s4p.exists() and "aotizhongxin" in by_h.study.unique():
        s4 = pd.read_csv(s4p)
        rows = []
        g = by_h[by_h.study == "aotizhongxin"]
        for m, old in S4_NAMES.items():
            if m not in g.method.unique():
                continue
            r = {"model": mlabel(m)}
            for h in (1, 6, 24):
                r[f"RMSE h={h} (§4)"] = float(s4[(s4.method == old) & (s4.h == h)].rmse.iloc[0])
                r[f"RMSE h={h} (here)"] = float(g[(g.method == m) & (g.h == h)].rmse.iloc[0])
            rows.append(r)
        section("Reproduction of real_series §4 (Aotizhongxin)", 3)
        emit(md(pd.DataFrame(rows)))
        emit("")

    # PMM theoretical MSE
    thp = R / "part1_pmm_theory.csv"
    if thp.exists() and thp.stat().st_size > 10:
        th = pd.read_csv(thp)
        section("PMM: empirical MSE against the theoretical MSE (paper Eqs. 18–24)", 3)
        rows = []
        for study in [s for s in SERIES if s in th.study.unique()]:
            t = th[th.study == study]
            for h in (1, 2, 3, 6, 12, 24):
                r = t[t.h == h]
                if len(r):
                    r = r.iloc[0]
                    rows.append({"series": study, "h": h, "theoretical MSE": r.theory_mse,
                                 "empirical MSE": r.emp_mse, "ratio": r.emp_mse / r.theory_mse,
                                 "mean predictive variance": r.emp_mean_pred_var})
        emit(md(pd.DataFrame(rows), {"h": "{:d}", "ratio": "{:.2f}"}, "{:.4f}"))
        emit("")
        tex(pd.DataFrame(rows), "PMM: empirical vs theoretical MSE", {"h": "{:d}", "ratio": "{:.2f}"}, "{:.4f}")
        figure_theory(th, F)

    pr = pd.read_csv(R / "part1_paired.csv")
    section("Paired CRPS differences against the PMM (per-origin mean over h; 95 % bootstrap CI)", 3)
    t = pr[pr.b == "pmm|raw"].copy()
    t["model"] = [mlabel(a) for a in t.a]
    t["CI"] = [f"[{lo:+.4f}, {hi:+.4f}]" for lo, hi in zip(t.ci_lo, t.ci_hi)]
    emit(md(t[["study", "model", "n_origins", "mean_diff", "CI", "frac_a_better"]],
            {"n_origins": "{:d}", "mean_diff": "{:+.4f}", "frac_a_better": "{:.2f}"}))
    emit("")
    figure_part1(by_h, F)


def figure_part1(by_h: pd.DataFrame, F: Path) -> None:
    studies = [s for s in SERIES if s in by_h.study.unique()]
    fig, axes = plt.subplots(1, len(studies), figsize=(4.2 * len(studies), 3.4))
    axes = np.atleast_1d(axes)
    for ax, study in zip(axes, studies):
        meths = [m for m in part1_methods(by_h, study) if m != "pmm_default|raw"]
        styles = ["-", "-", "--", ":", "-."]
        for i, m in enumerate(meths):
            g = by_h[(by_h.study == study) & (by_h.method == m)].sort_values("h")
            col = C[0] if m.startswith("pmc_") else C[6] if m.startswith("hmc") else \
                C[1] if m.startswith("pmm") else C[2] if m.startswith("ar1") else INK2
            ax.plot(g.h, g.crps, color=col, lw=2, ls=styles[i % len(styles)], label=mlabel(m))
        style(ax)
        ax.set_title(SERIES[study], fontsize=9, color=INK)
        ax.set_xlabel("horizon h (steps)", fontsize=8, color=INK2)
        ax.set_ylabel("CRPS", fontsize=8, color=INK2)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(F / "part1_crps_by_horizon.png", dpi=150)
    plt.close(fig)


def figure_theory(th: pd.DataFrame, F: Path) -> None:
    studies = [s for s in SERIES if s in th.study.unique()]
    fig, axes = plt.subplots(1, len(studies), figsize=(4.2 * len(studies), 3.2))
    axes = np.atleast_1d(axes)
    for ax, study in zip(axes, studies):
        t = th[th.study == study].sort_values("h")
        ax.plot(t.h, t.theory_mse, color=C[1], lw=2, label="theoretical MSE (TheoreticalMSE, Y entry)")
        ax.plot(t.h, t.emp_mse, color=C[0], lw=0, marker="o", ms=4, label="empirical MSE of the PMM forecasts")
        style(ax)
        ax.set_title(SERIES[study], fontsize=9, color=INK)
        ax.set_xlabel("horizon h", fontsize=8, color=INK2)
        ax.set_ylabel("MSE", fontsize=8, color=INK2)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(F / "pmm_theoretical_vs_empirical_mse.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# PMM fits, gating on clean data, quadrature
# ---------------------------------------------------------------------------

def fits_tables(R: Path) -> None:
    fits = pd.read_csv(R / "fits.csv")
    pmp = R / "pmm_fits.csv"
    if pmp.exists():
        pm = pd.read_csv(pmp)
        section("pmmforecast fits: default starts against the AR(1)-embedding start", 3)
        real = pm[~pm.name.str.startswith("fix_")].copy()
        ar = fits[(fits.family == "ar1") & (fits.fit == "raw")].set_index("case")
        real["AR(1) loglik"] = [ar.loc[n, "loglik"] if n in ar.index else np.nan for n in real.name]
        real["default loglik"] = -real.default_nll_std - real.n_fit * np.log(real.sd)
        real["ar1-start loglik"] = -real.ar1_start_nll_std - real.n_fit * np.log(real.sd)
        t = real[["name", "n_fit", "default loglik", "ar1-start loglik", "AR(1) loglik", "chosen_fit",
                  "default_c", "c", "trA", "detA", "spectral_radius"]]
        emit(md(t, {"n_fit": "{:d}", "default loglik": "{:.1f}", "ar1-start loglik": "{:.1f}",
                    "AR(1) loglik": "{:.1f}"}, "{:.4f}"))
        emit("")
        fx = pm[pm.name.str.startswith("fix_")]
        if len(fx):
            d = (fx.default_nll_std - fx.ar1_start_nll_std)
            emit(f"Fixture series ({len(fx)} fits): the AR(1)-embedding start reaches a higher likelihood in "
                 f"{int((d > 1e-6).sum())}, the default starts in {int((d < -1e-6).sum())} "
                 f"(median gain of the better fit over the default {np.median(np.maximum(d, 0)):.2f} nats).")
            emit("")
    section("Gating on the clean real series (flags at α = 1e-3 in the test part) and its cost", 3)
    by = pd.read_csv(R / "part1_all_h.csv").set_index(["study", "method"])
    rows = []
    for _, r in fits[fits.case.str.endswith("__clean") & ~fits.case.str.startswith("fix_")
                     & fits.fit.isin(["raw", "robust"])].iterrows():
        study = r.case.split("__")[0]
        fam = r.family
        base = f"{fam}|raw" if r.fit == "raw" else f"{fam}|robust"
        gated = f"{fam}|{r.fit}+gate"
        if (study, gated) not in by.index:
            continue
        rows.append({"series": study, "model": FAM_LABEL.get(fam, fam), "fit": r.fit,
                     "test rows flagged": r.gate_test_n_flag,
                     "per 1000": r.gate_test_false_per_1000,
                     "CRPS": by.loc[(study, base), "crps"] if (study, base) in by.index else np.nan,
                     "CRPS gated": by.loc[(study, gated), "crps"],
                     "masked train rows": r.get("mask_n_flag", np.nan)})
    t = pd.DataFrame(rows)
    if len(t):
        t["change %"] = 100 * (t["CRPS gated"] / t["CRPS"] - 1)
        emit(md(t, {"test rows flagged": "{:.0f}", "per 1000": "{:.2f}", "masked train rows": "{:.0f}",
                    "change %": "{:+.1f}"}, "{:.4f}"))
        emit("")
    section("Quadrature: nodes chosen per fitted copula model and quad_error (see Library problems, P1)", 3)
    q = fits[fits.gap_nodes.notna() & (fits.quad_check_64 > 0)].copy()
    if len(q):
        q["study"] = [c.split("__")[0] for c in q.case]
        for c in ("quad_error_obs", "quad_limit_obs", "quad_error_gated", "quad_limit_gated"):
            if c not in q.columns:
                q[c] = np.nan
        q["qe_ratio"] = np.fmax(q.quad_error_obs / q.quad_limit_obs, q.quad_error_gated / q.quad_limit_gated)
        q["unconv"] = (q.quad_check > 0.01) | (q.quad_check_tail > 0.05)
        agg = q.groupby(["study", "family"]).agg(fits=("fit", "size"), G64=("gap_nodes", lambda s: int((s == 64).sum())),
                                                  G128=("gap_nodes", lambda s: int((s == 128).sum())),
                                                  G256=("gap_nodes", lambda s: int((s == 256).sum())),
                                                  unconverged=("unconv", "sum"),
                                                  check_64=("quad_check_64", "max"),
                                                  check_G=("quad_check", "max"),
                                                  tail_G=("quad_check_tail", "max"),
                                                  ratio_64=("quad_ratio_64", "max"),
                                                  ratio_G=("quad_ratio", "max"),
                                                  cond_ratio_G=("qe_ratio", "max")).reset_index()
        emit("G: first of 64 / 128 / 256 nodes whose predictive means and sds (first, middle and last origin) "
             "change by less than 1 % of the predictive sd when G is doubled (`check`; `unconverged`: > 1 % at "
             "G = 256). `tail_G`: change of the node-law 2.5 / 97.5 % quantiles at the chosen G. `ratio`: "
             "pmcprg's `GapPosterior.quad_error` / WARNING limit of the same forecast chains (recorded, not "
             "required). Worst values over the fits; `cond_ratio_G`: quad_error / limit of the whole "
             "conditioning series, observed and gated (– : no missing row).")
        emit("")
        emit(md(agg, {"fits": "{:d}", "G64": "{:d}", "G128": "{:d}", "G256": "{:d}", "unconverged": "{:d}",
                      "check_64": "{:.1e}", "check_G": "{:.1e}", "tail_G": "{:.1e}", "ratio_64": "{:.2g}",
                      "ratio_G": "{:.2g}", "cond_ratio_G": "{:.2g}"}))
        emit("")
    wp = R / "pmcprg_warnings.csv"
    if wp.exists() and wp.stat().st_size > 5:
        w = pd.read_csv(wp)
        section("pmcprg WARNINGs collected during the campaign", 3)
        w["study"] = [c.split("__")[0] for c in w.case.fillna("mote20__select")]
        w["step"] = [s.split(" ", 1)[-1] for s in w.step.fillna("")]
        t = w.groupby(["kind", "study", "step", "template"], dropna=False).agg(
            records=("count", "sum"), max_value=("max_value", "max")).reset_index()
        t["template"] = [s[:90] for s in t.template]
        emit(md(t, {"records": "{:d}", "max_value": "{:.2e}"}))
        emit("")


# ---------------------------------------------------------------------------
# Part 2
# ---------------------------------------------------------------------------

def part2(R: Path, F: Path) -> None:
    sp = R / "crosscheck_summary.csv"
    if not sp.exists():
        return
    s = pd.read_csv(sp)
    section("Part 2 — Gaussian cross-check")
    t = s[["tuple", "abcde", "d_minus_bc", "e_minus_bc", "Y_is_AR1", "c", "gamma2", "c2", "ll_pmm",
           "ll_pmc_ar1map", "ll_ar1_closed", "ll_pmc_minus_pmm"]]
    emit(md(t, {"ll_pmm": "{:.4f}", "ll_pmc_ar1map": "{:.4f}", "ll_ar1_closed": "{:.4f}",
                "ll_pmc_minus_pmm": "{:+.2e}", "d_minus_bc": "{:+.4f}", "e_minus_bc": "{:+.4f}",
                "gamma2": "{:.4f}", "c2": "{:.4f}", "c": "{:.3f}"}))
    emit("")
    tex(t.drop(columns=["abcde"]), "Cross-check: log-likelihoods (N = 2000)",
        {"ll_pmc_minus_pmm": "{:+.2e}", "ll_pmm": "{:.3f}", "ll_pmc_ar1map": "{:.3f}", "ll_ar1_closed": "{:.3f}"})
    t = s[["tuple", "max_abs_mean_pmc_vs_ar1", "max_abs_sd_pmc_vs_ar1", "max_abs_quantile_pmc_vs_gauss",
           "max_abs_mean_pmm_vs_ar1", "max_abs_sd_pmm_vs_ar1", "max_abs_theoryMSE_vs_pmm_var"]]
    emit("Forecasts at origins 50, 500, 1000, 2000, h = 1…20 (data scale y = 2 + 0.5 Y): largest absolute "
         "differences.")
    emit("")
    emit(md(t, default="{:.1e}"))
    emit("")
    m = pd.read_csv(R / "crosscheck_mse.csv")
    t = m[m.h.isin([1, 2, 3, 5, 10])][["tuple", "h", "theory_pmm", "one_minus_c2h", "theory_ar1map", "emp_pmm",
                                        "se_pmm", "emp_pmc_ar1map", "se_pmc_ar1map"]]
    emit("Monte-Carlo MSE (40 000-step path, 975 origins; model scale):")
    emit("")
    emit(md(t, {"h": "{:d}"}, "{:.4f}"))
    emit("")
    ip = R / "crosscheck_identifiability.csv"
    if ip.exists():
        i = pd.read_csv(ip)
        if "nll_y_only_unstandardised" in i.columns:   # the MLE row: NLL of the same (unstandardised) path
            i["nll_y_only"] = i.nll_y_only_unstandardised.fillna(i.nll_y_only)
        emit("Identifiability: Y-only negative log-likelihood of pmmforecast on the 'generic' path "
             "(`neg_log_likelihood_y_only(tuple, y)`) for tuples with the same (c, tr A, det A), and the "
             "tuple returned by `estimate_pmm_mle_y_only(y, n_restarts=4, seed=0)`:")
        emit("")
        cols = [c for c in ("tuple", "abcde", "c", "trA", "detA", "nll_y_only") if c in i.columns]
        emit(md(i[cols], {"nll_y_only": "{:.6f}"}, "{:.4f}"))
        emit("")


# ---------------------------------------------------------------------------
# Part 3
# ---------------------------------------------------------------------------

P3_VARIANTS = ["true", "true+gate", "raw", "raw+gate", "robust", "robust+gate", "hampel+gate", "clean"]


def part3(R: Path, F: Path) -> None:
    fx = pd.read_csv(R / "part3_fixtures_by_h.csv")
    fxc = pd.read_csv(R / "part3_fixtures_by_case.csv")
    section("Part 3 — robust forecasting")
    section("Simulated fixtures (report/erroneous_data), spikes of 6 sd", 3)
    for study in [s for s in FIXTURES if s in fx.study.unique()]:
        g = fxc[fxc.study == study]
        fam = [m.split("|")[0] for m in g.method.unique() if m.split("|")[0] not in ("ar1", "persistence", "pmm",
                                                                                     "pmm_default")][0]
        meths = [f"{fam}|{v}" for v in P3_VARIANTS] + ["pmm|raw", "ar1|raw", "ar1|raw+gate",
                                                       "ar1|robust+gate", "ar1|clean", "persistence|raw"]
        meths = [m for m in meths if m in g.method.unique()]
        rates = sorted(g.rate.unique())
        rows = []
        for m in meths:
            r = {"method": mlabel(m)}
            for rate in rates:
                cr = g[(g.method == m) & (g.rate == rate)]
                r[f"CRPS {rate:.0%}"] = cr.crps.mean()
                h1 = fx[(fx.study == study) & (fx.method == m) & (fx.rate == rate) & (fx.h == 1)]
                r[f"CRPS h=1 {rate:.0%}"] = float(h1.crps.iloc[0]) if len(h1) else np.nan
            if 0.0 in rates:   # PMM on the clean series = the rate-0 PMM of the same replicate
                pass
            rows.append(r)
        t = pd.DataFrame(rows)
        # win counts over replicates at 5 %
        emit(f"**{FIXTURES[study]}** — CRPS averaged over h = 1…10, 99 origins and the replicates; "
             "and at h = 1:")
        emit("")
        emit(md(t))
        emit("")
        tex(t, f"{FIXTURES[study]}: CRPS (mean over h and replicates) and at h = 1")
    sl = pd.read_csv(R / "part3_fixtures_spike_last_h1.csv")
    rows = []
    for study in [s for s in FIXTURES if s in sl.study.unique()]:
        g = sl[(sl.study == study) & (sl.rate > 0)]
        fam = "hmc_in_K2" if study.startswith("hmc") else "pmc_state_K2"
        for m in [f"{fam}|raw", f"{fam}|raw+gate", f"{fam}|robust+gate", f"{fam}|hampel+gate", f"{fam}|true",
                  f"{fam}|true+gate", "pmm|raw", "ar1|raw", "ar1|raw+gate", "ar1|robust+gate", "persistence|raw"]:
            for rate in sorted(g.rate.unique()):
                x = g[(g.method == m) & (g.rate == rate)].set_index("spike_last")
                if True in x.index and False in x.index:
                    rows.append({"fixture": FIXTURES[study], "rate": f"{rate:.0%}", "method": mlabel(m),
                                 "n spike last": int(x.loc[True, "n"]), "CRPS h=1, spike last": x.loc[True, "crps"],
                                 "CRPS h=1, otherwise": x.loc[False, "crps"]})
    if rows:
        emit("One-step forecasts whose last conditioning row is a spike, against the others (h = 1):")
        emit("")
        emit(md(pd.DataFrame(rows), {"n spike last": "{:d}"}))
        emit("")
    real = pd.read_csv(R / "part3_real_by_case.csv")
    rh = pd.read_csv(R / "part3_real_by_h.csv")
    for study, title in (("aotizhongxin", "Beijing Aotizhongxin with injected spikes (6 sd), 3 seeds per rate"),
                         ("mote20", "Intel Lab mote 20 as recorded (3 suspect readings of −38.4 °C in the training part)")):
        g = real[real.study == study]
        if not len(g):
            continue
        section(title, 3)
        g = g.copy()
        g["variant"] = [c.split("__")[1] for c in g.case]
        fams = sorted({m.split("|")[0] for m in g.method.unique()} - {"ar1", "persistence", "pmm", "pmm_default"})
        meths = [f"{f}|{v}" for f in fams for v in ("raw", "raw+gate", "robust", "robust+gate", "hampel+gate")]
        meths += ["pmm|raw", "ar1|raw", "ar1|raw+gate", "ar1|robust+gate", "persistence|raw"]
        meths = [m for m in meths if m in g.method.unique()]
        cols = sorted(g.rate.unique())
        H = int(rh[rh.study == study].h.max())
        rows = []
        for m in meths:
            r = {"method": mlabel(m)}
            for rate in cols:
                x = g[(g.method == m) & (g.rate == rate)]
                lab = "clean" if rate == 0 and study == "aotizhongxin" else ("recorded" if study == "mote20" and
                                                                              (x.case.str.endswith("raw").any()) else
                                                                              f"{rate:.0%}")
                if study == "mote20":
                    for v in ("clean", "raw"):
                        xv = g[(g.method == m) & (g.variant == v)]
                        r[f"CRPS {'suspect removed' if v == 'clean' else 'as recorded'}"] = xv.crps.mean() if len(xv) else np.nan
                        hv = rh[(rh.study == study) & (rh.method == m) & (rh.variant == v) & (rh.h == 1)]
                        r[f"RMSE h=1 {'suspect removed' if v == 'clean' else 'as recorded'}"] = \
                            float(np.sqrt(hv.mse.mean())) if len(hv) else np.nan
                    break
                r[f"CRPS {lab}"] = x.crps.mean() if len(x) else np.nan
                h1 = rh[(rh.study == study) & (rh.method == m) & (rh.rate == rate) & (rh.h == 1)]
                r[f"CRPS h=1 {lab}"] = float(h1.crps.mean()) if len(h1) else np.nan
            rows.append(r)
        t = pd.DataFrame(rows)
        emit(f"CRPS averaged over h = 1…{H} and the origins (and the seeds), and at h = 1"
             + (" (the copula PMC ran on seed 0 only: compare it in the seed-0 table below):"
                if study == "aotizhongxin" else ":"))
        emit("")
        emit(md(t))
        emit("")
        tex(t, title)
        if study == "aotizhongxin":
            s0 = g[g.case.str.endswith("__clean") | g.case.str.endswith("_s0")]
            rows = []
            for m in meths:
                r = {"method": mlabel(m)}
                for rate in cols:
                    lab = "clean" if rate == 0 else f"{rate:.0%}"
                    x = s0[(s0.method == m) & (s0.rate == rate)]
                    r[f"CRPS {lab}"] = x.crps.mean() if len(x) else np.nan
                    h1 = rh[(rh.study == study) & (rh.method == m) & (rh.rate == rate) & (rh.h == 1)
                            & (rh.variant.str.endswith("_s0") | (rh.variant == "clean"))]
                    r[f"CRPS h=1 {lab}"] = float(h1.crps.mean()) if len(h1) else np.nan
                rows.append(r)
            t0 = pd.DataFrame(rows)
            emit("Seed 0 only (every method ran on the same contaminated series):")
            emit("")
            emit(md(t0))
            emit("")
            tex(t0, "Aotizhongxin with injected spikes, seed 0")
    fits = pd.read_csv(R / "fits.csv")
    section("Robust fits on the real series", 3)
    rf = fits[~fits.case.str.startswith("fix_") & fits.fit.isin(["robust", "hampel"])].copy()
    cols = ["case", "family", "fit", "n_fits", "converged", "mask_n_flag", "mask_detect", "mask_false_per_1000",
            "prescreen_detect", "degenerate", "gap_nodes", "quad_check", "seconds"]
    cols = [c for c in cols if c in rf.columns]
    emit(md(rf[cols], {"n_fits": "{:.0f}", "mask_n_flag": "{:.0f}", "mask_detect": "{:.2f}",
                       "mask_false_per_1000": "{:.2f}", "prescreen_detect": "{:.2f}", "gap_nodes": "{:.0f}",
                       "quad_check": "{:.1e}", "seconds": "{:.0f}"}))
    emit("")
    section("Robust fits on the fixtures (means over replicates)", 3)
    ff = fits[fits.case.str.startswith("fix_") & fits.fit.isin(["raw", "robust", "hampel"]) &
              (fits.family != "ar1")].copy()
    ff["fixture"] = [c.split("__")[0][4:] for c in ff.case]
    ff["rate"] = [float(c.split("__")[1].split("_")[0][1:]) for c in ff.case]
    agg = ff.groupby(["fixture", "rate", "fit"]).agg(
        n=("case", "size"), n_fits=("n_fits", "mean"), converged=("converged", lambda s: float(np.mean(s == True))),  # noqa: E712
        mask_detect=("mask_detect", "mean"), mask_false_per_1000=("mask_false_per_1000", "mean"),
        gate_test_detect=("gate_test_detect", "mean"), gate_test_false_per_1000=("gate_test_false_per_1000", "mean"),
        degenerate=("degenerate", "mean")).reset_index()
    emit(md(agg, {"n": "{:d}", "rate": "{:.0%}", "n_fits": "{:.1f}", "converged": "{:.2f}"}, "{:.3f}"))
    emit("")
    ar = fits[(fits.family == "ar1")].copy()
    ar["study"] = [c.split("__")[0] for c in ar.case]
    ar["variant"] = [c.split("__")[1] for c in ar.case]
    section("AR(1) with innovation gating: flags", 3)
    real_ar = ar[~ar.study.str.startswith("fix_")]
    cols = [c for c in ("case", "fit", "phi", "sigma2", "n_fits", "converged", "gate_train_n_flag", "mask_n_flag",
                        "mask_detect", "gate_test_n_flag", "gate_test_detect", "gate_test_false_per_1000")
            if c in real_ar.columns]
    emit(md(real_ar[cols], {"n_fits": "{:.0f}", "gate_train_n_flag": "{:.0f}", "mask_n_flag": "{:.0f}",
                            "gate_test_n_flag": "{:.0f}", "phi": "{:.4f}", "sigma2": "{:.4g}"}))
    emit("")
    figure_part3(fxc, rh, F)


def figure_part3(fxc: pd.DataFrame, rh: pd.DataFrame, F: Path) -> None:
    studies = [s for s in FIXTURES if s in fxc.study.unique()]
    fig, axes = plt.subplots(1, len(studies) + 1, figsize=(4.3 * (len(studies) + 1), 3.5))
    for ax, study in zip(axes, studies):
        g = fxc[fxc.study == study]
        fam = "hmc_in_K2" if study.startswith("hmc") else "pmc_state_K2"
        series = [(f"{fam}|raw", C[0], "-", "o"), (f"{fam}|raw+gate", C[0], "--", "s"),
                  (f"{fam}|hampel+gate", C[2], "-", "^"), ("pmm|raw", C[1], "-", "D"),
                  ("ar1|raw", C[3], "-", "v"), ("ar1|robust+gate", C[3], "--", "x")]
        for m, col, ls, mk in series:
            t = g[g.method == m].groupby("rate").crps.mean()
            if len(t):
                ax.plot(100 * t.index, t.values, color=col, ls=ls, marker=mk, ms=6, lw=2, label=mlabel(m))
        ref = g[g.method == f"{fam}|clean"].groupby("rate").crps.mean()
        ax.axhline(ref.mean(), color=INK2, lw=1, ls=":", label="clean fit on the clean series")
        style(ax)
        ax.set_title(FIXTURES[study], fontsize=9, color=INK)
        ax.set_xlabel("spike rate (%)", fontsize=8, color=INK2)
        ax.set_ylabel("CRPS (mean over h = 1…10)", fontsize=8, color=INK2)
        ax.set_xticks([0, 1, 5])
        ax.legend(fontsize=6.5, frameon=False)
    ax = axes[-1]
    g = rh[(rh.study == "aotizhongxin") & (rh.variant.str.endswith("_s0") | (rh.variant == "clean"))]
    fam = "pmc_pair_K3"
    for m, col, ls in ((f"{fam}|raw", C[0], "-"), (f"{fam}|raw+gate", C[0], "--"), (f"{fam}|hampel+gate", C[2], "-"),
                       ("pmm|raw", C[1], "-"), ("ar1|robust+gate", C[3], "--")):
        t = g[(g.method == m) & (g.rate == 0.05)].groupby("h").crps.mean()
        if len(t):
            ax.plot(t.index, t.values, color=col, ls=ls, lw=2, label=mlabel(m))
    t = g[(g.method == f"{fam}|raw") & (g.rate == 0)].groupby("h").crps.mean()
    ax.plot(t.index, t.values, color=INK2, lw=1, ls=":", label="PMC pair K=3 on the clean series")
    style(ax)
    ax.set_title("Aotizhongxin, 5 % spikes of 6 sd (seed 0)", fontsize=9, color=INK)
    ax.set_xlabel("horizon h (hours)", fontsize=8, color=INK2)
    ax.set_ylabel("CRPS", fontsize=8, color=INK2)
    ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(F / "part3_robust_forecasting.png", dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--figures", type=Path, default=HERE / "figures")
    args = ap.parse_args(argv)
    args.figures.mkdir(parents=True, exist_ok=True)
    R = args.results
    campaign = (R / "run_info.json").exists() and (R / "part1_by_h.csv").exists()
    if campaign:
        info = json.loads((R / "run_info.json").read_text())
        emit(f"Forecasting study — {info['date']}, {info['n_records']} score rows, wall {info['wall_s'] / 60:.1f} "
             f"min, sum of pmcprg task times {info['sum_task_s'] / 60:.1f} min, pmmforecast jobs "
             f"{(info.get('pmm_sum_s') or 0) / 60:.1f} min ({info['python']}, numpy {info['numpy']}, "
             f"{info['machine']}).")
        emit(f"Quantile CRPS (200 levels) against the exact CRPS: {json.dumps(info.get('crps_quantile_vs_exact'))}")
        emit(f"Rows with quantiles inconsistent with the node law (pmcprg P3; q_check > 0.25 on each horizon's "
             f"own grid): {json.dumps(info.get('quantile_inconsistent_rows'))}; on the reference nodes: "
             f"{json.dumps(info.get('quantile_inconsistent_rows_reference_nodes'))}; "
             f"statistics {json.dumps(info.get('quantile_check_stats'))}")
        emit(f"pmcprg WARNINGs: {json.dumps(info.get('pmcprg_warnings'))}")
        part1(R, args.figures)
        fits_tables(R)
    else:
        emit("No campaign results in this folder (run_forecasting.py): parts 1 and 3 skipped.")
    part2(R, args.figures)
    if campaign and (R / "part3_fixtures_by_h.csv").exists():
        part3(R, args.figures)
    (args.results / "tables.md").write_text("\n".join(OUT) + "\n")
    (args.results / "tables.tex").write_text("% generated by report/forecasting/summarise.py\n" + "\n".join(TEX))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
