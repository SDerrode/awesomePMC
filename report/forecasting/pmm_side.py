"""
pmm_side.py — every use of pmmforecast in the forecasting study (forecasting).

pmmforecast (PyPI ``pmmforecast``, import package ``prg``, AGPL) is the companion
code of Escudier, Abdelkefi, Fernandes & Pieczynski, "Forecasting with Pairwise
Gaussian Markov Models", CMCSI 2023 (arXiv 2402.07532): a stationary 1-D Gaussian
pairwise Markov model (PMM) with a continuous hidden state X and five parameters
(a, b, c, d, e). It is NOT a dependency of awesomePMC and nothing of it is copied
here: this script imports it, and runs in a Python where it is installed. The
pmcprg side (``run_forecasting.py``, ``cross_check.py``, run with the awesomePMC
interpreter) exchanges CSV/JSON files with it. Without pmmforecast the script
prints a message and exits with status 3; the drivers then skip the PMM columns.

Subcommands
-----------
jobs --dir D [--workers W] [--force]
    For every ``D/<job>.json`` (+ ``D/<job>.csv``, column ``y``, NaN allowed)
    written by ``run_forecasting.py``:

    * fit the PMM by pmmforecast's Y-only Kalman MLE
      (``prg.inference.estimation.estimate_pmm_mle_y_only``, Nelder–Mead) on
      ``y[first observed : n_train]``, twice: with pmmforecast's default starts
      (``n_restarts``, ``seed``) and from one start whose law of Y is the AR(1)
      of the series' lag-1 autocorrelation (``fit_pmm``); the higher
      likelihood is used (``fit = best`` in the output), the default fit is
      also forecast (``fit = default``). pmmforecast has no missing-value
      handling, so interior NaN of the training part are filled by linear
      interpolation (choice of this study);
    * filter the whole series with ``PMMFilter1D.filter_observations`` (the
      same linear fill; causal for every origin, see ``_origin_forecast``) and,
      at every origin o, forecast from the last observed row L < o:
      E[Z_{L+k} | Y_{≤L}] = A^k (E[X_L | Y_{≤L}], y_L) (``power_a_generator``)
      and Cov[Z_{L+k} | Y_{≤L}] from S_{L+1} (``filter_observations``) and
      ``PMMPredictor1D.var_predict_generator`` — the recursions of
      ``PMMPredictor1D.predict_generator``, which only drives pmmforecast's own
      simulator; k = 1 … H + (o − 1 − L), the last H kept;
    * the closed-form prediction MSE of the paper (Eqs. 18–24,
      ``TheoreticalMSE.compute_th_mse_for_pmm``): its ``th_mse`` output is the
      MSE of X; the MSE of Y is the (Y, Y) entry of the covariance table it
      returns, used here, after ``theory_n0`` observations;
    * write ``<job>.pmm.csv`` (fit, origin, h, mean, sd on the data scale) and
      ``<job>.pmm.json`` (fits, likelihoods, invariants, theoretical MSE).

crosscheck --out D
    Part 2 (Gaussian cross-check): PMM tuples, simulated paths, Y-only
    likelihoods, forecasts and theoretical MSEs, and the identifiability
    checks; ``cross_check.py`` compares them with pmcprg.

Usage (pmmforecast in a separate venv; OMP_NUM_THREADS=1)::

    <venv>/bin/python report/forecasting/pmm_side.py jobs --dir report/out/forecasting/pmm_jobs
    <venv>/bin/python report/forecasting/pmm_side.py crosscheck --out report/out/forecasting/crosscheck
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

try:  # pmmforecast
    import prg  # noqa: F401
    from prg.analysis import TheoreticalMSE
    from prg.inference import PMMFilter1D, PMMPredictor1D
    from prg.inference.estimation import (
        estimate_pmm_mle_y_only,
        neg_log_likelihood_y_only,
        standardise,
    )
    from prg.models import ParamPMM, PMMSimulator1D
    HAVE_PMM = True
except ImportError:  # pragma: no cover - reported to the user
    HAVE_PMM = False

#: Exit status when pmmforecast is not importable (the drivers skip the PMM part).
NO_PMM_STATUS = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fill_linear(y: np.ndarray) -> np.ndarray:
    """Linear interpolation of interior NaN; NaN beyond the ends get the nearest value."""
    y = np.asarray(y, float).copy()
    fin = np.isfinite(y)
    if fin.all():
        return y
    t = np.arange(y.size)
    y[~fin] = np.interp(t[~fin], t[fin], y[fin])
    return y


def invariants(p) -> dict:
    """(c, tr A, det A): the law of Y depends on the tuple through these only.

    γ_Y(1) = c and, by Cayley–Hamilton, γ_Y(k) = tr A γ_Y(k−1) − det A γ_Y(k−2)
    for k ≥ 2 (γ_Y(0) = 1, and γ_Y(2) = tr A · c − det A because A⁻¹Q₂ = Q₁):
    Y is a Gaussian ARMA(2, 1) with AR polynomial 1 − tr A L + det A L².
    """
    A = p.get_a_matrix()
    return {"c": float(p.get_c()), "trA": float(np.trace(A)), "detA": float(np.linalg.det(A))}


def acf_y(p, kmax: int) -> np.ndarray:
    """γ_Y(k), k = 0..kmax, from the invariants."""
    inv = invariants(p)
    g = np.empty(kmax + 1)
    g[0] = 1.0
    if kmax >= 1:
        g[1] = inv["c"]
    for k in range(2, kmax + 1):
        g[k] = inv["trA"] * g[k - 1] - inv["detA"] * g[k - 2]
    return g


def predictive_moments(p, e_L: float, z_L: float, S_next: np.ndarray, nsteps: int):
    """Y-mean and Y-variance of Z_{L+k} | Y_{≤L}, k = 1..nsteps (model scale).

    Mean: A^k (e_L, z_L) (``ParamPMM.power_a_generator``); covariance:
    S_{L+1} then ``PMMPredictor1D.var_predict_generator`` (S ← A S Aᵀ + BBᵀ),
    as in ``PMMPredictor1D.predict_generator``.
    """
    means = np.empty(nsteps)
    for k, Ak in enumerate(p.power_a_generator(nsteps)):
        means[k] = Ak[1, 0] * e_L + Ak[1, 1] * z_L
    var = np.empty(nsteps)
    var[0] = S_next[1, 1]
    if nsteps > 1:
        pred = PMMPredictor1D(p, p, p, n_samples=2, start_predict_index=0)
        for k, Sk in enumerate(pred.var_predict_generator(nsteps - 1, S_next)):
            var[k + 1] = Sk[1, 1]
    return means, var


def theoretical_y_mse(p, n0: int, H: int) -> np.ndarray:
    """Paper Eqs. 18–24 (``TheoreticalMSE``): MSE of Y at horizons 1..H after n0 observations.

    ``compute_th_mse_for_pmm`` returns (th_mse, th_mse_s); th_mse is the MSE of
    X, th_mse_s[n] the covariance Cov[Z_n | Y_0..Y_{n0−1}] in the prediction
    region (n ≥ n0), whose (Y, Y) entry is the MSE of the Y forecast.
    TheoreticalMSE builds an (n0 + H)² Toeplitz table in its constructor, so
    n0 is kept moderate (the filter variance converges geometrically).
    """
    th = TheoreticalMSE(p, n_samples=n0 + H, start_predict_index=n0 - 1)
    _, th_s = th.compute_th_mse_for_pmm()
    return np.array([th_s[n0 + k - 1, 1, 1] for k in range(1, H + 1)])


# ---------------------------------------------------------------------------
# Forecast jobs
# ---------------------------------------------------------------------------

def _origin_forecast(p, e, S, z, first: int, fin: np.ndarray, o: int, H: int):
    """Forecast from the last observed row L < o (a trailing gap is skipped exactly).

    The global linear fill is causal here: every filled row ≤ L lies in a gap
    closed by an observed row ≤ L, and the rows after L are never used.
    """
    L = int(np.nonzero(fin[:o])[0][-1])
    g = o - 1 - L
    Lz = L - first
    S_next = S[Lz + 1] if Lz + 1 < len(S) else None
    if S_next is None:  # origin past the end of the filtered series (not used here)
        raise ValueError("origin beyond the filtered series")
    mean, var = predictive_moments(p, float(e[Lz]), float(z[Lz]), S_next, H + g)
    return mean[g:], var[g:], g


def ar1_start(z: np.ndarray) -> tuple:
    """A start inside the PMM family whose law of Y is the AR(1) of the lag-1 autocorrelation r.

    (a, b, c, d, e) = (0.5, 0.3, r, 0.3 r, 0.3 r): pmmforecast's default a and b,
    c = r, and d = e = bc, so Y is exactly an AR(1)(r) (cross_check.py); falls back
    to (0, 0, r, 0, 0) (X white noise) if that tuple is not a valid PMM.
    """
    r = float(np.clip(np.mean(z[1:] * z[:-1]), -0.999, 0.999))
    for x0 in ((0.5, 0.3, r, 0.3 * r, 0.3 * r), (0.0, 0.0, r, 0.0, 0.0)):
        try:
            ParamPMM(*x0)
            return x0
        except Exception:
            continue
    raise ValueError("no valid AR(1) start")


def fit_pmm(y_train: np.ndarray, z_train: np.ndarray, n_restarts: int, seed: int) -> dict:
    """Two calls of ``estimate_pmm_mle_y_only``; the forecasts use the higher likelihood.

    * ``default`` — pmmforecast's defaults: first start (0.5, 0.3, 0.1, 0.4, 0.2),
      ``n_restarts − 1`` uniform draws in [−0.7, 0.7]^5 (``seed``);
    * ``ar1_start`` — one start at :func:`ar1_start`. On persistent series
      (lag-1 autocorrelation ≥ 0.99) the default starts end tens of nats below
      the AR(1) sub-model (README, "pmmforecast issues").
    """
    out = {}
    t = time.perf_counter()
    p, res = estimate_pmm_mle_y_only(y_train, n_restarts=n_restarts, seed=seed)
    out["default"] = {"params": p, "success": bool(res.success), "nfev": int(res.nfev),
                      "nll": float(neg_log_likelihood_y_only(np.array(p.get_abcde()), z_train)),
                      "seconds": time.perf_counter() - t}
    x0 = ar1_start(z_train)
    t = time.perf_counter()
    p, res = estimate_pmm_mle_y_only(y_train, x0=x0, n_restarts=1)
    out["ar1_start"] = {"params": p, "success": bool(res.success), "nfev": int(res.nfev),
                        "nll": float(neg_log_likelihood_y_only(np.array(p.get_abcde()), z_train)),
                        "seconds": time.perf_counter() - t, "x0": list(x0),
                        "x0_nll": float(neg_log_likelihood_y_only(np.array(x0), z_train))}
    return out


def run_job(spec_path: str, force: bool = False) -> dict:
    logging.disable(logging.WARNING)
    spec_path = Path(spec_path)
    spec = json.loads(spec_path.read_text())
    out_csv = spec_path.with_suffix(".pmm.csv")
    out_json = spec_path.with_suffix(".pmm.json")
    if out_json.exists() and out_csv.exists() and not force:
        return {"name": spec["name"], "cached": True}
    t0 = time.perf_counter()
    y = pd.read_csv(spec_path.with_suffix(".csv"))["y"].to_numpy(float)
    fin = np.isfinite(y)
    first = int(np.argmax(fin))
    n_train, H = int(spec["n_train"]), int(spec["H"])
    y_train = fill_linear(y[first:n_train])
    m, s = float(y_train.mean()), float(y_train.std())
    z_train = (y_train - m) / s
    assert np.allclose(standardise(y_train), z_train)
    fits = fit_pmm(y_train, z_train, int(spec.get("n_restarts", 4)), int(spec.get("seed", 0)))
    t_fit = time.perf_counter() - t0
    best = min(fits, key=lambda k: fits[k]["nll"])
    params = fits[best]["params"]
    # Whole series: same standardisation (training mean and sd), same fill.
    z = (fill_linear(y[first:]) - m) / s
    rows = []
    gaps = []
    filt = {}
    for lab in ("best", "default"):
        p = params if lab == "best" else fits["default"]["params"]
        e, v, S = PMMFilter1D(None, p, len(z)).filter_observations(z)
        filt[lab] = (e, v, S)
        for o in spec["origins"]:
            mean, var, g = _origin_forecast(p, e, S, z, first, fin, int(o), H)
            if lab == "best":
                gaps.append(g)
            for k in range(H):
                rows.append({"fit": lab, "origin": int(o), "h": k + 1, "mean": m + s * mean[k],
                             "sd": s * float(np.sqrt(max(var[k], 0.0)))})
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    e, v, S = filt["best"]
    n0 = int(spec.get("theory_n0", 500))
    th = theoretical_y_mse(params, n0, H)
    # Predictive variance at the last origin with no trailing gap (steady-state filter).
    o_ss = next((int(o) for o in reversed(spec["origins"]) if fin[int(o) - 1]), None)
    ss = None
    if o_ss is not None:
        _, var_ss, _ = _origin_forecast(params, e, S, z, first, fin, o_ss, H)
        ss = (s * s * var_ss).tolist()
    A = params.get_a_matrix()
    info = {
        "name": spec["name"], "n_fit": int(y_train.size), "n_filled_train": int((~fin[first:n_train]).sum()),
        "mean": m, "sd": s, "chosen_fit": best, "abcde": [float(x) for x in params.get_abcde()],
        **invariants(params), "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(A)))),
        "nll_std": fits[best]["nll"], "loglik_data": float(-fits[best]["nll"] - y_train.size * np.log(s)),
        "n_restarts": int(spec.get("n_restarts", 4)), "seed": int(spec.get("seed", 0)),
        "theory_n0": n0, "theory_y_mse": (s * s * th).tolist(), "steady_pred_var": ss,
        "filter_var_last": float(v[-1]), "filter_var_n0": float(v[min(n0 - 1, len(v) - 1)]),
        "max_trailing_gap": int(max(gaps)) if gaps else 0,
        "seconds_fit": t_fit, "seconds": time.perf_counter() - t0,
        "pmmforecast_version": getattr(prg, "__version__", "?"),
    }
    for lab, f in fits.items():
        info[f"{lab}_abcde"] = [float(x) for x in f["params"].get_abcde()]
        info[f"{lab}_nll_std"] = f["nll"]
        info[f"{lab}_c"], info[f"{lab}_trA"], info[f"{lab}_detA"] = (
            invariants(f["params"])[k] for k in ("c", "trA", "detA"))
        info[f"{lab}_success"], info[f"{lab}_nfev"] = f["success"], f["nfev"]
        info[f"{lab}_seconds"] = f["seconds"]
    info["ar1_start_x0"] = fits["ar1_start"]["x0"]
    info["ar1_start_x0_nll_std"] = fits["ar1_start"]["x0_nll"]
    out_json.write_text(json.dumps(info, indent=1))
    return {"name": spec["name"], "cached": False, "seconds": info["seconds"]}


def cmd_jobs(args) -> int:
    d = Path(args.dir)
    specs = sorted(p for p in d.glob("*.json") if not p.name.endswith(".pmm.json"))
    # Longest series first (better load balance).
    specs.sort(key=lambda p: -json.loads(p.read_text())["n_train"])
    print(f"pmm_side jobs: {len(specs)} job(s) in {d}, {args.workers} worker(s)", flush=True)
    t0 = time.perf_counter()
    if args.workers <= 1:
        for sp in specs:
            r = run_job(str(sp), args.force)
            print("  ", r, flush=True)
    else:
        with ProcessPoolExecutor(args.workers) as ex:
            futs = {ex.submit(run_job, str(sp), args.force): sp for sp in specs}
            for f in as_completed(futs):
                print("  ", f.result(), flush=True)
    print(f"pmm_side jobs done in {time.perf_counter() - t0:.1f} s", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Part 2: Gaussian cross-check
# ---------------------------------------------------------------------------

#: PMM tuples of the cross-check. d = bc or e = bc: Y is a Gaussian AR(1)(c);
#: "generic" and "hmm": Y is ARMA(2, 1), no PMC describes it exactly.
XC_TUPLES = {
    "d_eq_bc": (0.8, 0.5, 0.7, 0.35, 0.6),
    "e_eq_bc": (0.8, 0.5, 0.7, 0.6, 0.35),
    "generic": (0.8, 0.5, 0.7, 0.5, 0.65),
    "hmm": (0.9, 0.8, 0.9 * 0.8 ** 2, 0.9 * 0.8, 0.9 * 0.8),
}
XC_N = 2000            # short path: likelihoods and forecasts at fixed origins
XC_ORIGINS = (50, 500, 1000, 2000)
XC_H = 20
XC_MC_N = 40000        # long path: Monte-Carlo MSE
XC_MC_H = 10
XC_MC_STEP = 40
XC_MC_FIRST = 1000
XC_N0 = 500


def _simulate(p, n: int, seed: int) -> np.ndarray:
    sim = PMMSimulator1D(p, n_samples=n)
    sim.set_seed(seed)
    return np.array([tuple(t) for t in sim.simulate_generator()], dtype=float)


def _same_law_tuple(p, b_new: float):
    """Another valid tuple with the same (c, tr A, det A), at a different b (least squares)."""
    from scipy.optimize import least_squares
    target = invariants(p)
    c = target["c"]

    def resid(v):
        a, d, e = v
        den = 1.0 - b_new * b_new
        return [(a + c - b_new * (d + e)) / den - target["trA"], (a * c - d * e) / den - target["detA"]]

    a0, _, _, d0, e0 = p.get_abcde()
    best = None
    for start in ([a0, d0, e0], [a0, e0, d0], [0.5, 0.3, 0.3], [0.5, -0.3, 0.3]):
        r = least_squares(resid, start, xtol=1e-15, ftol=1e-15, gtol=1e-15)
        a, d, e = r.x
        try:
            q = ParamPMM(float(a), b_new, c, float(d), float(e))
        except Exception:
            continue
        if np.max(np.abs(r.fun)) < 1e-12:
            best = q
            break
    return best


def cmd_crosscheck(args) -> int:
    logging.disable(logging.WARNING)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    info: dict = {"pmmforecast_version": getattr(prg, "__version__", "?"), "tuples": {}}
    for ti, (name, tup) in enumerate(XC_TUPLES.items()):
        p = ParamPMM(*tup)
        rec = {"abcde": list(tup), **invariants(p), "d_minus_bc": tup[3] - tup[1] * tup[2],
               "e_minus_bc": tup[4] - tup[1] * tup[2], "acf": acf_y(p, 30).tolist()}
        xy = _simulate(p, XC_N, seed=1000 + ti)
        pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1]}).to_csv(out / f"{name}_short.csv", index=False)
        y = xy[:, 1]
        rec["nll_y_only"] = float(neg_log_likelihood_y_only(np.array(tup), y))
        e, v, S = PMMFilter1D(None, p, len(y)).filter_observations(y)
        rows = []
        for o in XC_ORIGINS:
            L = o - 1
            if L + 1 < len(S):
                S_next = S[L + 1]
            else:  # one step past the end: S_{L+1} = Var[X_L|.] (α1, α3)(α1, α3)ᵀ + BBᵀ
                a1, a3 = p.get_alpha_1(), p.get_alpha_3()
                S_next = v[L] * np.array([[a1 * a1, a1 * a3], [a1 * a3, a3 * a3]]) + p.get_bbt()
            mean, var = predictive_moments(p, float(e[L]), float(y[L]), S_next, XC_H)
            th = theoretical_y_mse(p, o, XC_H) if o <= 1000 else None
            for k in range(XC_H):
                rows.append({"origin": o, "h": k + 1, "mean": mean[k], "var": var[k],
                             "th_y_mse": float(th[k]) if th is not None else np.nan})
        pd.DataFrame(rows).to_csv(out / f"{name}_short_fc.csv", index=False)
        rec["theory_y_mse_n0"] = theoretical_y_mse(p, XC_N0, XC_MC_H).tolist()
        # Monte-Carlo path.
        xy = _simulate(p, XC_MC_N, seed=2000 + ti)
        y = xy[:, 1]
        pd.DataFrame({"y": y}).to_csv(out / f"{name}_mc.csv", index=False)
        e, v, S = PMMFilter1D(None, p, len(y)).filter_observations(y)
        rows = []
        for o in range(XC_MC_FIRST, XC_MC_N - XC_MC_H + 1, XC_MC_STEP):
            mean, var = predictive_moments(p, float(e[o - 1]), float(y[o - 1]), S[o], XC_MC_H)
            for k in range(XC_MC_H):
                rows.append({"origin": o, "h": k + 1, "mean": mean[k], "var": var[k],
                             "y": float(y[o + k])})
        pd.DataFrame(rows).to_csv(out / f"{name}_mc_fc.csv", index=False)
        info["tuples"][name] = rec
        print(f"crosscheck: {name} done", flush=True)

    # Identifiability: tuples with the same (c, tr A, det A) have the same Y-only likelihood.
    p = ParamPMM(*XC_TUPLES["generic"])
    y = pd.read_csv(out / "generic_short.csv")["y"].to_numpy()
    a, b, c, d, e = XC_TUPLES["generic"]
    others = {"generic": p, "sign_flip(b,d,e)": ParamPMM(a, -b, c, -d, -e),
              "swap(d,e)": ParamPMM(a, b, c, e, d)}
    for b_new in (0.2, 0.7):
        q = _same_law_tuple(p, b_new)
        if q is not None:
            others[f"same_law_b={b_new}"] = q
    ident = []
    for lab, q in others.items():
        ident.append({"tuple": lab, "abcde": [float(x) for x in q.get_abcde()], **invariants(q),
                      "nll_y_only": float(neg_log_likelihood_y_only(np.array(q.get_abcde()), y))})
    t0 = time.perf_counter()
    est, res = estimate_pmm_mle_y_only(y, n_restarts=4, seed=0)
    ident.append({"tuple": "Y-only MLE (n_restarts=4)", "abcde": [float(x) for x in est.get_abcde()],
                  **invariants(est), "nll_y_only": float(neg_log_likelihood_y_only(
                      np.array(est.get_abcde()), standardise(y))),
                  "nll_y_only_unstandardised": float(neg_log_likelihood_y_only(np.array(est.get_abcde()), y)),
                  "seconds": time.perf_counter() - t0})
    info["identifiability"] = ident
    (out / "pmm_crosscheck.json").write_text(json.dumps(info, indent=1))
    print("crosscheck: written", out / "pmm_crosscheck.json", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("jobs")
    j.add_argument("--dir", required=True)
    j.add_argument("--workers", type=int, default=1)
    j.add_argument("--force", action="store_true")
    c = sub.add_parser("crosscheck")
    c.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if not HAVE_PMM:
        print("pmm_side: pmmforecast (import package 'prg') is not installed in this Python "
              f"({sys.executable}); the PMM part of the forecasting study is skipped. "
              "Install pmmforecast in a separate venv and pass its interpreter to "
              "run_forecasting.py / cross_check.py with --pmm-python.", file=sys.stderr)
        return NO_PMM_STATUS
    return cmd_jobs(args) if args.cmd == "jobs" else cmd_crosscheck(args)


if __name__ == "__main__":
    sys.exit(main())
