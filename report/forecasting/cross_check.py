"""
cross_check.py — part 2 of the forecasting study: pmcprg and pmmforecast on the
same Gaussian law of Y (forecasting).

Mapping (derivation in README.md, part 2). In pmmforecast's stationary PMM
(zero mean, unit variances, Γ of (X_n, Y_n, X_{n+1}, Y_{n+1}) built from
(a, b, c, d, e), A = Q₂Q₁⁻¹) the autocovariance of Y is γ(1) = c and
γ(k) = tr A · γ(k−1) − det A · γ(k−2) for k ≥ 2: Y is ARMA(2, 1) and its law
depends on (c, tr A, det A) only. γ(2) = c² ⇔ (d − bc)(e − bc) = 0, and then
γ(k) = c^k: Y is a Gaussian AR(1) of coefficient c. A pmcprg PMC with Gaussian
margins and Gaussian copulas has a Gaussian law of (y_n, y_{n+1}) only if all
its (i, j) components are the same bivariate Gaussian, and then Y is an AR(1).
So the two packages describe the same law of Y exactly when (d − bc)(e − bc) = 0
and the PMC has identical regimes: margins N(μ, σ²), Gaussian copulas of
ρ = c (Kendall τ = (2/π) arcsin c) on every pair, any prior; y = μ + σ Y.

Checks, on paths simulated by pmmforecast (``pmm_side.py crosscheck``):

* log-likelihoods: pmmforecast's Y-only Kalman NLL (+ the Jacobian N log σ),
  pmcprg's ``classify`` log-likelihood of the identical-regime model, and the
  closed-form AR(1) likelihood (``rs_common._ar1_nll``);
* h-step forecasts at fixed origins: pmmforecast (filter + A^k, covariance
  recursion), pmcprg ``forecast`` (quadrature grid), AR(1) closed form;
* the prediction MSE: ``TheoreticalMSE`` (Y entry), 1 − c^{2h}, pmcprg's
  predictive variance, and Monte-Carlo MSEs on a 40 000-step path;
* for the non-AR(1) tuples ("generic", "hmm") the same with the AR(1) of
  coefficient c: the loss of the closest pmcprg description.

Usage (repository root)::

    PYTHONPATH=. .venv/bin/python report/forecasting/cross_check.py --pmm-python <venv>/bin/python
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fc_common as fc  # noqa: E402

from pmcprg.pmc import classify, forecast  # noqa: E402

REPO = HERE.parents[1]
#: Data scale of the cross-check: y = MU + SIGMA · Y (arbitrary, exercises the mapping).
MU, SIGMA = 2.0, 0.5
#: Conditioning window of the Monte-Carlo forecasts (exact for a Markov Y).
MC_WINDOW = 200


def run_pmm(pmm_python: str | None, out: Path, force: bool) -> bool:
    js = out / "pmm_crosscheck.json"
    if js.exists() and not force:
        return True
    if not pmm_python:
        print("cross_check: no --pmm-python (or PMMFORECAST_PYTHON) given and no pmmforecast output in "
              f"{out}: the pmmforecast side of the cross-check is skipped.")
        return False
    r = subprocess.run([pmm_python, "-B", str(HERE / "pmm_side.py"), "crosscheck", "--out", str(out)],
                       cwd=REPO)
    if r.returncode != 0:
        print(f"cross_check: pmm_side.py exited with status {r.returncode}; skipped.")
        return False
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pmm-python", default=os.environ.get("PMMFORECAST_PYTHON"))
    ap.add_argument("--out", type=Path, default=REPO / "report/out/forecasting/crosscheck")
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)
    if not run_pmm(args.pmm_python, args.out, args.force):
        return 0
    t0 = time.perf_counter()
    info = json.loads((args.out / "pmm_crosscheck.json").read_text())
    summ, fcs, mse = [], [], []
    for name, rec in info["tuples"].items():
        c = rec["c"]
        ar1 = {"mu": MU, "phi": c, "sigma2": SIGMA ** 2 * (1 - c * c)}
        M = fc.ar1_model(ar1)
        acf = np.array(rec["acf"])
        exact = abs(rec["d_minus_bc"]) < 1e-12 or abs(rec["e_minus_bc"]) < 1e-12
        # --- likelihoods on the short path
        y = pd.read_csv(args.out / f"{name}_short.csv")["y"].to_numpy()
        N = y.size
        yd = MU + SIGMA * y
        ll_pmm = -rec["nll_y_only"] - N * math.log(SIGMA)
        ll_pmc = float(classify(M, yd)[2])
        ll_ar1 = -float(fc.rc._ar1_nll(np.array([MU, math.atanh(c), math.log(ar1["sigma2"])]), yd, np.ones(N - 1)))
        ar1_fit = fc.rc.ar1_fit(yd)
        row = {"tuple": name, "abcde": " ".join(f"{v:.4g}" for v in rec["abcde"]),
               "d_minus_bc": rec["d_minus_bc"], "e_minus_bc": rec["e_minus_bc"], "Y_is_AR1": exact,
               "c": c, "trA": rec["trA"], "detA": rec["detA"], "gamma2": acf[2], "c2": c * c,
               "N": N, "ll_pmm": ll_pmm, "ll_pmc_ar1map": ll_pmc, "ll_ar1_closed": ll_ar1,
               "ll_pmc_minus_pmm": ll_pmc - ll_pmm, "ll_ar1_minus_pmm": ll_ar1 - ll_pmm,
               "ll_ar1_mle": ar1_fit["loglik"], "ar1_mle_phi": ar1_fit["phi"]}
        # --- forecasts at fixed origins (full prefix)
        pf = pd.read_csv(args.out / f"{name}_short_fc.csv")
        dm = dsd = dq = dvar_th = dmm = dms = 0.0
        for o, g in pf.groupby("origin"):
            g = g.sort_values("h")
            H = len(g)
            f = forecast(M, yd[:o], H, quantiles=fc.QLEVELS)
            k = np.arange(1, H + 1)
            m_ar1 = MU + c ** k * (yd[o - 1] - MU)
            s_ar1 = SIGMA * np.sqrt(1 - c ** (2 * k))
            m_pmm = MU + SIGMA * g["mean"].to_numpy()
            s_pmm = SIGMA * np.sqrt(g["var"].to_numpy())
            dm = max(dm, float(np.max(np.abs(f.mean - m_ar1))))
            dsd = max(dsd, float(np.max(np.abs(f.sd - s_ar1))))
            dq = max(dq, float(np.max(np.abs(f.quantile_values - fc.gauss_quantiles(m_ar1, s_ar1)))))
            dmm = max(dmm, float(np.max(np.abs(m_pmm - m_ar1))))
            dms = max(dms, float(np.max(np.abs(s_pmm - s_ar1))))
            th = g["th_y_mse"].to_numpy()
            if np.all(np.isfinite(th)):
                dvar_th = max(dvar_th, float(np.max(np.abs(th - g["var"].to_numpy()))))
            for kk in range(H):
                fcs.append({"tuple": name, "origin": int(o), "h": kk + 1, "pmm_mean": m_pmm[kk], "pmm_sd": s_pmm[kk],
                            "pmc_mean": f.mean[kk], "pmc_sd": f.sd[kk], "ar1_mean": m_ar1[kk], "ar1_sd": s_ar1[kk],
                            "th_y_mse": th[kk]})
        row.update({"max_abs_mean_pmc_vs_ar1": dm, "max_abs_sd_pmc_vs_ar1": dsd,
                    "max_abs_quantile_pmc_vs_gauss": dq, "max_abs_mean_pmm_vs_ar1": dmm,
                    "max_abs_sd_pmm_vs_ar1": dms, "max_abs_theoryMSE_vs_pmm_var": dvar_th})
        # --- Monte-Carlo MSE (model scale)
        ymc = pd.read_csv(args.out / f"{name}_mc.csv")["y"].to_numpy()
        mf = pd.read_csv(args.out / f"{name}_mc_fc.csv")
        ydm = MU + SIGMA * ymc
        Hm = int(mf.h.max())
        pmc_mean = {}
        for o in sorted(mf.origin.unique()):
            f = forecast(M, ydm[o - MC_WINDOW:o], Hm, quantiles=(0.5,))
            pmc_mean[int(o)] = (f.mean - MU) / SIGMA
        mf["pmc_mean"] = [pmc_mean[int(o)][int(h) - 1] for o, h in zip(mf.origin, mf.h)]
        th_pmm = np.array(rec["theory_y_mse_n0"])
        for h, g in mf.groupby("h"):
            e_pmm = (g.y - g["mean"]) ** 2
            e_pmc = (g.y - g.pmc_mean) ** 2
            mse.append({"tuple": name, "h": int(h), "n_origins": len(g),
                        "theory_pmm": th_pmm[int(h) - 1],
                        "theory_ar1map": 1 - 2 * c ** h * acf[int(h)] + c ** (2 * h),
                        "one_minus_c2h": 1 - c ** (2 * h),
                        "emp_pmm": float(e_pmm.mean()), "se_pmm": float(e_pmm.std() / math.sqrt(len(g))),
                        "emp_pmc_ar1map": float(e_pmc.mean()), "se_pmc_ar1map": float(e_pmc.std() / math.sqrt(len(g))),
                        "mean_pmm_pred_var": float(g["var"].mean())})
        summ.append(row)
        print(f"cross_check: {name}: ll pmc−pmm {row['ll_pmc_minus_pmm']:+.2e}, ar1−pmm "
              f"{row['ll_ar1_minus_pmm']:+.2e}; max |mean pmc − ar1| {dm:.1e}, |sd| {dsd:.1e}")
    pd.DataFrame(summ).to_csv(args.results / "crosscheck_summary.csv", index=False)
    pd.DataFrame(mse).to_csv(args.results / "crosscheck_mse.csv", index=False)
    pd.DataFrame(fcs).to_csv(args.out / "crosscheck_forecasts.csv", index=False)
    ident = pd.DataFrame(info["identifiability"])
    ident["abcde"] = [" ".join(f"{v:.4f}" for v in t) for t in ident.abcde]
    ident.to_csv(args.results / "crosscheck_identifiability.csv", index=False)
    (args.results / "crosscheck_info.json").write_text(json.dumps(
        {"pmmforecast_version": info.get("pmmforecast_version"), "MU": MU, "SIGMA": SIGMA,
         "mc_window": MC_WINDOW, "seconds_pmcprg_side": time.perf_counter() - t0}, indent=1))
    print(f"cross_check: written to {args.results} ({time.perf_counter() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
