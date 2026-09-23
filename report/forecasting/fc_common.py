"""
fc_common.py — data, models, scores and tasks of the forecasting study (forecasting).

pmcprg side only (run with the awesomePMC interpreter); everything that imports
pmmforecast lives in ``pmm_side.py``. Data preparation, split, fits and baselines
are those of ``report/real_series`` (``rs_common``), so that the Aotizhongxin
numbers are comparable with its section 4; the contamination and the Hampel
pre-screen are those of ``report/erroneous_data``.

Every task is a pure function of its spec (seeds fixed in the spec), so the
results do not depend on the number of worker processes.
"""

from __future__ import annotations

import logging
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "report" / "real_series")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rs_common as rc  # noqa: E402

from pmcprg.pmc import PMCModel, classify, flag_outliers, forecast, ice, robust_estimate, simulate  # noqa: E402

DEFAULT_DATA = rc.DEFAULT_DATA
logger = logging.getLogger("forecasting")

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

#: Quantile levels of the CRPS: mid-points of 200 equal bins, CRPS ≈ (2/M) Σ_i
#: QS_τi (quantile decomposition of the CRPS, Gneiting & Raftery 2007, eq. 22).
TAUS = (np.arange(200) + 0.5) / 200
#: Central intervals scored for coverage: nominal level → (lower, upper) quantile levels.
COVERAGE = {50: (0.25, 0.75), 80: (0.10, 0.90), 95: (0.025, 0.975)}
#: Every quantile level requested from a predictive law.
QLEVELS = tuple(sorted(set(TAUS.tolist()) | {0.025, 0.10, 0.25, 0.5, 0.75, 0.90, 0.975}))
_QIDX = {q: i for i, q in enumerate(QLEVELS)}
_TAU_IDX = np.array([_QIDX[t] for t in TAUS.tolist()])

#: Per-row level of flag_outliers (the library default, as in report/erroneous_data).
ALPHA = 1e-3
#: Hampel pre-screen of report/erroneous_data (centred window of 11, 4 MADs).
HAMPEL_HALF, HAMPEL_T = 5, 4.0
#: Spike size (multiples of the series sd) of the contaminated cases.
SPIKE_K = 6
#: pmmforecast Y-only MLE: Nelder–Mead starts (first from pmmforecast's default x0,
#: then uniform draws in [−0.7, 0.7]^5 with ``PMM_SEED``).
PMM_RESTARTS = 4
PMM_SEED = 0
#: TheoreticalMSE is evaluated after this many observations (its constructor
#: builds an (n0 + H)² table; the filter variance has converged long before).
PMM_THEORY_N0 = 500

#: Real series: horizon H, origin step, train fraction, pmcprg models
#: (None: chosen by BIC on the training part, see ``select_models``).
REAL = {
    "aotizhongxin": {"H": 24, "step": 24, "train_frac": 0.75,
                     "models": [("pmc_pair", 3), ("hmc_in", 3)],
                     "label": "Beijing Aotizhongxin, ln PM2.5, 1 h"},
    "tsnh4": {"H": 24, "step": 12, "train_frac": 0.75,
              "models": [("pmc_state", 2), ("hmc_in", 3)],
              "label": "tsNH4 complete, ln NH4, 10 min"},
    "mote20": {"H": 24, "step": 60, "train_frac": 0.75, "models": None,
               "label": "Intel Lab mote 20, temperature °C, 30 s"},
}
#: Mote-20 window: epochs 1..28800 (the 10-day window of the Intel Lab study extracts).
MOTE_ID, MOTE_EPOCHS = 20, 28800
#: Resolution of the mote temperature (°C) and seed of its dequantisation jitter.
MOTE_STEP, MOTE_DEQUANT_SEED = 0.0098, 11
#: Candidate models of the mote-20 BIC choice.
MOTE_CANDIDATES = [(k, K) for k in ("hmc_in", "pmc_state", "pmc_pair") for K in (2, 3)]

#: Simulated fixtures of report/erroneous_data.
FIXTURES = {"hmc_in_gauss_k2": ("hmc_in", 2), "pmc_gauss_k2": ("pmc_state", 2)}
FIX_N_TRAIN, FIX_N_TEST, FIX_H, FIX_STEP = 2000, 1000, 10, 10
FIX_RATES = (0.0, 0.01, 0.05)
FIX_REPS = 10
REAL_RATES = (0.01, 0.05)
REAL_SEEDS = 3


# ---------------------------------------------------------------------------
# Seeds, contamination, Hampel
# ---------------------------------------------------------------------------

def seed_of(*parts) -> int:
    """zlib.crc32 of the labels (the seeding rule of report/erroneous_data)."""
    return zlib.crc32("|".join(str(p) for p in parts).encode())


def isolated_positions(N: int, rate: float, rng: np.random.Generator,
                       allowed: np.ndarray | None = None) -> np.ndarray:
    """round(rate·N) isolated positions in 1..N−2 (no two adjacent), among ``allowed``.

    Same rule as ``report/erroneous_data/run_study.py::_spikes``; with
    ``allowed`` (observed rows of a real series) only those rows are drawn.
    """
    want = int(round(rate * N))
    chosen = np.zeros(N, dtype=bool)
    k = 0
    for c in rng.permutation(np.arange(1, N - 1)):
        if k == want:
            break
        if (allowed is not None and not allowed[c]) or chosen[c - 1] or chosen[c + 1]:
            continue
        chosen[c] = True
        k += 1
    return chosen


def hampel_nan(y: np.ndarray, half: int = HAMPEL_HALF, t: float = HAMPEL_T) -> np.ndarray:
    """Hampel identifier |y_n − med| > t · 1.4826 · MAD on a centred window, NaN-aware.

    Equal to ``report/erroneous_data``'s ``hampel`` on a series without NaN
    (checked in ``run_forecasting.py --quick``); NaN rows are never flagged and
    are ignored in the window statistics.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    y = np.asarray(y, float)
    pad = np.pad(y, half, mode="reflect")
    win = sliding_window_view(pad, 2 * half + 1)
    with np.errstate(all="ignore"), _warnings_off():
        med = np.nanmedian(win, axis=1)
        mad = 1.4826 * np.nanmedian(np.abs(win - med[:, None]), axis=1)
    out = np.abs(y - med) > t * np.maximum(mad, 1e-12)
    return out & np.isfinite(y)


class _warnings_off:
    def __enter__(self):
        import warnings
        self._c = warnings.catch_warnings()
        self._c.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *a):
        self._c.__exit__(*a)


# ---------------------------------------------------------------------------
# Cases: a series as observed (Y), its scoring truth, split and origins
# ---------------------------------------------------------------------------

_MOTE_CACHE: dict = {}


def load_mote(data_dir: str) -> dict:
    """Mote-20 temperature, epochs 1..MOTE_EPOCHS, dequantised.

    The readings are multiples of 0.0098 °C (the sensor's resolution: the
    suspect −38.4 °C readings are the zero count, T = −38.4 + 0.0098 · count)
    and 29 % of consecutive readings are equal: as with Beijing's integer
    PM2.5 (real_series), a copula PMC then puts a state on an atom and its
    likelihood is unbounded (every copula fit of the smoke run was
    degenerate). The fitted series is ``Y + U(−δ/2, δ/2)``, δ = 0.0098, fixed
    seed; forecasts are scored against the readings as recorded (``Y_rec``,
    suspect readings and absent epochs NaN).
    """
    key = (data_dir, MOTE_ID)
    if key not in _MOTE_CACHE:
        df = pd.read_csv(Path(data_dir) / f"intel_lab/output/mote{MOTE_ID}_epochs.csv",
                         usecols=["epoch", "suspect", "Y_raw", "Y"])
        df = df[df.epoch <= MOTE_EPOCHS]
        jit = np.random.default_rng([MOTE_DEQUANT_SEED, MOTE_ID]).uniform(-0.5, 0.5, len(df)) * MOTE_STEP
        _MOTE_CACHE[key] = {"Y": df.Y.to_numpy(float) + jit, "Y_raw": df.Y_raw.to_numpy(float) + jit,
                            "Y_rec": df.Y.to_numpy(float), "suspect": df.suspect.to_numpy(int) == 1}
    return _MOTE_CACHE[key]


def real_base(series: str, data_dir: str) -> dict:
    """Clean observed series (what is fitted and conditioned on) and scoring truth."""
    if series == "aotizhongxin":
        bj = rc.load_beijing(data_dir, "aotizhongxin", False)
        return {"Y": bj["Y"], "truth": bj["Y_lograw"]}
    if series == "tsnh4":
        # The NaN of tsNH4 were inserted by the imputeTS authors; the complete
        # series is the real record: it is used whole here (no gap).
        d = rc.load_tsnh4(data_dir, False)
        return {"Y": d["Y_true"], "truth": d["Y_true"]}
    if series == "mote20":
        m = load_mote(data_dir)
        return {"Y": m["Y"], "truth": m["Y_rec"]}
    raise ValueError(series)


def split(N: int, H: int, step: int, train_frac: float) -> tuple[int, np.ndarray]:
    """First train_frac for fitting; origins every ``step`` in the rest (rs_common.task_forecast)."""
    n_train = int(train_frac * N)
    return n_train, np.arange(n_train, N - H, step)


def case_ids() -> list[str]:
    ids = [f"{s}__clean" for s in REAL]
    ids.append("mote20__raw")
    ids += [f"aotizhongxin__spikes_r{r:.2f}_k{SPIKE_K}_s{s}" for r in REAL_RATES for s in range(REAL_SEEDS)]
    ids += [f"fix_{f}__r{r:.2f}_k{SPIKE_K}_rep{rep}" for f in FIXTURES for r in FIX_RATES
            for rep in range(FIX_REPS)]
    return ids


def build_case(case: str, data_dir: str, quick: bool = False) -> dict:
    """Y (observed, NaN allowed), truth, clean Y, spike mask, n_train, origins, H.

    ``quick`` shortens the real series (smoke run only).
    """
    series, variant = case.split("__")
    if series.startswith("fix_"):
        name = series[4:]
        rate = float(variant.split("_")[0][1:])
        rep = int(variant.split("rep")[1])
        model = PMCModel(ROOT / "pmcprg" / "pmc" / "models" / f"{name}.toml")
        n_train, n_test = (500, 300) if quick else (FIX_N_TRAIN, FIX_N_TEST)
        N = n_train + n_test
        X, Y = simulate(model, N, seed=seed_of("sim", name, rep))
        spikes = (isolated_positions(N, rate, np.random.default_rng(seed_of("cont", name, SPIKE_K, rate, rep)))
                  if rate > 0 else np.zeros(N, dtype=bool))
        Yc = Y.copy()
        Yc[spikes] += SPIKE_K * Y.std()
        origins = np.arange(n_train, N - FIX_H, FIX_STEP)
        return {"case": case, "series": series, "variant": variant, "Y": Yc, "truth": Y, "Y_clean": Y,
                "X": X, "spikes": spikes, "n_train": n_train, "origins": origins, "H": FIX_H,
                "rate": rate, "fixture": name}
    cfg = REAL[series]
    base = real_base(series, data_dir)
    Y0, truth = base["Y"].copy(), base["truth"].copy()
    spikes = np.zeros(len(Y0), dtype=bool)
    Y = Y0.copy()
    rate = 0.0
    if variant == "raw":                       # mote 20 as recorded (suspect readings kept)
        m = load_mote(data_dir)
        Y = m["Y_raw"].copy()
        spikes = m["suspect"].copy()
    elif variant.startswith("spikes"):
        rate = float(variant.split("_")[1][1:])
        seed = int(variant.split("_s")[-1])
        spikes = isolated_positions(len(Y0), rate,
                                    np.random.default_rng(seed_of("cont", series, SPIKE_K, rate, seed)),
                                    allowed=np.isfinite(Y0))
        Y[spikes] += SPIKE_K * float(np.nanstd(Y0))
    if quick:
        n = 3000
        Y, Y0, truth, spikes = Y[:n], Y0[:n], truth[:n], spikes[:n]
    H = cfg["H"]
    step = cfg["step"] * (4 if quick else 1)
    n_train, origins = split(len(Y), H, step, cfg["train_frac"])
    return {"case": case, "series": series, "variant": variant, "Y": Y, "truth": truth, "Y_clean": Y0,
            "spikes": spikes, "n_train": n_train, "origins": origins, "H": H, "rate": rate}


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

_SQPI = 1.0 / math.sqrt(math.pi)


def crps_gauss(y, m, s) -> np.ndarray:
    """Closed-form CRPS of N(m, s²) at y, elementwise."""
    y, m, s = np.broadcast_arrays(np.asarray(y, float), np.asarray(m, float), np.asarray(s, float))
    z = (y - m) / s
    return s * (z * (2 * ndtr(z) - 1) + 2 * np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi) - _SQPI)


def _A(mu, var):
    s = np.sqrt(var)
    z = mu / s
    return 2 * s * np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi) + mu * (2 * ndtr(z) - 1)


def crps_gauss_mixture(y, w, mu, sd) -> np.ndarray:
    """Exact CRPS of Σ_k w_k N(mu_k, sd_k²) at y (Grimit et al. 2006); w: (n, K), mu/sd: (K,)."""
    y = np.asarray(y, float)[:, None]
    t1 = np.sum(w * _A(y - mu[None, :], sd[None, :] ** 2), axis=1)
    dm = mu[:, None] - mu[None, :]
    dv = sd[:, None] ** 2 + sd[None, :] ** 2
    t2 = 0.5 * np.einsum("ni,nj,ij->n", w, w, _A(dm, dv))
    return t1 - t2


def crps_quantiles(y, qv) -> np.ndarray:
    """CRPS from the predictive quantiles at TAUS (qv: (n, len(QLEVELS)))."""
    q = np.asarray(qv, float)[:, _TAU_IDX]
    y = np.asarray(y, float)[:, None]
    return 2.0 * np.mean(((y < q).astype(float) - TAUS[None, :]) * (q - y), axis=1)


def gauss_quantiles(m, s) -> np.ndarray:
    z = ndtri(np.array(QLEVELS))
    return np.asarray(m, float)[:, None] + np.asarray(s, float)[:, None] * z[None, :]


def rows_for(method: str, origin: int, fut: np.ndarray, mean, sd, qv, crps=None, extra=None) -> list[dict]:
    """Per-horizon score rows of one forecast (rows with a missing truth are skipped).

    crps: exact CRPS if given (Gaussian laws); otherwise from the quantiles.
    crps_gauss: the Gaussian CRPS of the predictive mean and sd (the score of
    real_series §4 for every model).
    """
    H = len(fut)
    ok = np.isfinite(fut)
    if not ok.any():
        return []
    mean = np.asarray(mean, float)
    sd = np.asarray(sd, float) if sd is not None else np.full(H, np.nan)
    crps_q = crps_quantiles(np.where(ok, fut, 0.0), qv)
    if crps is None:
        crps = crps_q
    cg = crps_gauss(np.where(ok, fut, 0.0), mean, np.where(np.isfinite(sd), sd, 1.0))
    out = []
    for k in np.nonzero(ok)[0]:
        r = {"method": method, "origin": int(origin), "h": int(k + 1), "y": float(fut[k]),
             "mean": float(mean[k]), "sd": float(sd[k]), "crps": float(crps[k]), "crps_q": float(crps_q[k]),
             "crps_gauss": float(cg[k]) if np.isfinite(sd[k]) else np.nan,
             "pit": float(np.interp(fut[k], qv[k], QLEVELS, left=0.0, right=1.0))}
        for lev, (lo, hi) in COVERAGE.items():
            r[f"c{lev}"] = bool(qv[k, _QIDX[lo]] <= fut[k] <= qv[k, _QIDX[hi]])
        r["w95"] = float(qv[k, _QIDX[0.975]] - qv[k, _QIDX[0.025]])
        if extra:
            r.update(extra)
        out.append(r)
    return out


def aggregate(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """RMSE, MAE, CRPS, Gaussian CRPS, coverage and 95 % width over the rows of each group."""
    g = df.groupby(keys, sort=False)
    out = g.agg(n=("y", "size"), mse=("err2", "mean"), mae=("abserr", "mean"), crps=("crps", "mean"),
                crps_gauss=("crps_gauss", "mean"), cov50=("c50", "mean"), cov80=("c80", "mean"),
                cov95=("c95", "mean"), w95=("w95", "mean")).reset_index()
    out.insert(len(keys) + 1, "rmse", np.sqrt(out.mse))
    return out


def add_errors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["err2"] = (df.y - df["mean"]) ** 2
    df["abserr"] = (df.y - df["mean"]).abs()
    return df


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def family(kind: str, K: int) -> str:
    return f"{kind}_K{K}"


def fit_kmeans(kind: str, K: int, Y: np.ndarray):
    """real_series fit: k-means warm start (seed 0), ICE "available" (rs_common)."""
    init, _ = rc.starting_model(kind, K, Y, 0, 1)
    return rc.estimate(init, Y, kind, "available", 0)


def kmeans_cfg(kind: str) -> dict:
    """ICE config whose ``init = "kmeans"`` reproduces rs_common's start 0 at every refit."""
    return {"fit_margins": True, "candidates": list(rc.COPULA_CANDIDATES), "selection_criterion": "mle",
            "margin_selection_rule": rc.margin_rule(kind), "init": "kmeans", "kmeans_seed": 0,
            "n_starts": 1, "max_iter": rc.ICE_MAX_ITER, "tol": rc.ICE_TOL,
            "missing_strategy": "available", "missing_seed": 1000}


def fixture_cfg(model: PMCModel) -> dict:
    """ICE config of report/erroneous_data (from the true model, Gaussian copulas for PMC)."""
    cfg = {"fit_margins": True}
    if model.variant.uses_copula:
        cfg["candidates"] = ["Gauss"]
    return cfg


def ar1_model(par: dict) -> PMCModel:
    """The Gaussian AR(1) as a pmcprg model: K = 2 identical regimes (part 2).

    Both state margins N(μ, σ²/(1 − φ²)), a Gaussian copula of Kendall τ =
    (2/π) arcsin φ on every pair: whatever the prior p, y_{n+1} | y_n, x_n,
    x_{n+1} has the same law, so Y is the AR(1). pmcprg requires K ≥ 2.
    """
    phi = float(np.clip(par["phi"], -0.999999, 0.999999))
    v = par["sigma2"] / (1 - phi ** 2)
    tau = 2.0 / math.pi * math.asin(phi)
    raw = {"model": {"name": "AR(1) as 2 identical regimes", "K": 2, "variant": "PMC", "N_default": 100},
           "prior": {"p": [[0.45, 0.05], [0.05, 0.45]]},
           "margins": [{"i": i, "dist": "norm", "params": {"loc": float(par["mu"]), "scale": float(math.sqrt(v))}}
                       for i in range(2)],
           "copulas": [{"i": i, "j": j, "name": "Gauss", "tau": tau} for i in range(2) for j in range(2)]}
    return PMCModel.from_dict(raw)


def ar1_flags(Y: np.ndarray, par: dict, alpha: float = ALPHA) -> np.ndarray:
    """Sequential innovation gating of a Gaussian AR(1), in closed form.

    Row n is tested with the predictive law N(μ + φ^k (y_L − μ), v(1 − φ^{2k}))
    from the last accepted row L (k = n − L; the stationary law for the first
    row), p = 2 min(Φ, 1 − Φ); a flagged row is treated as missing. This is
    ``flag_outliers`` on ``ar1_model(par)`` in exact arithmetic (checked in
    ``run_forecasting.py --quick``). The closed form is used because pmcprg's
    gap quadrature collapses when φ ≳ 0.998 (README, "Library problems"): a
    flagged row is a missing row for the filter of the next ones.
    """
    from scipy.stats import norm
    mu, phi, s2 = par["mu"], par["phi"], par["sigma2"]
    v = s2 / (1 - phi ** 2)
    Y = np.asarray(Y, float)
    flags = np.zeros(len(Y), dtype=bool)
    last = -1
    for n in np.nonzero(np.isfinite(Y))[0]:
        if last < 0:
            m, s = mu, math.sqrt(v)
        else:
            k = n - last
            m, s = mu + phi ** k * (Y[last] - mu), math.sqrt(max(v * (1 - phi ** (2 * k)), 1e-300))
        if 2 * norm.sf(abs(Y[n] - m) / s) < alpha:
            flags[n] = True
        else:
            last = n
    return flags


def ar1_robust(Y: np.ndarray, max_rounds: int = 10) -> tuple[dict, np.ndarray, int, bool]:
    """Flag-and-mask AR(1): the loop of robust_estimate with ``rs_common.ar1_fit``.

    Flags recomputed on the original Y at every round with the AR(1) of the
    previous fit (``ar1_flags``), flagged rows set to NaN for the next fit
    (ar1_fit handles gaps exactly); stops at a fixed point.
    """
    mask = np.zeros(len(Y), dtype=bool)
    par = rc.ar1_fit(Y)
    fits = 1
    for _ in range(max_rounds):
        f = ar1_flags(Y, par)
        if np.array_equal(f, mask):
            return par, mask, fits, True
        mask = f.copy()
        Ym = Y.copy()
        Ym[mask] = np.nan
        par = rc.ar1_fit(Ym)
        fits += 1
    f = ar1_flags(Y, par)
    return par, mask, fits, bool(np.array_equal(f, mask))


def quad_diff(model: PMCModel, Y: np.ndarray, origins, H: int, G: int, G_ref: int) -> float:
    """Worst relative change of the forecasts (mean and sd, in units of the G_ref sd)
    between G and G_ref quadrature nodes, at the first, middle and last origin;
    inf when a predictive sd collapses (< 1e-3 of the series sd)."""
    sd_y = float(np.nanstd(Y))
    worst = 0.0
    origins = list(origins)
    for o in sorted({origins[0], origins[len(origins) // 2], origins[-1]}):
        a = forecast(model, Y[:o], H, quantiles=(0.5,), gap_nodes=G)
        b = forecast(model, Y[:o], H, quantiles=(0.5,), gap_nodes=G_ref)
        if np.min(a.sd) < 1e-3 * sd_y or np.min(b.sd) < 1e-3 * sd_y:
            return math.inf
        worst = max(worst, float(np.max(np.abs(a.sd - b.sd) / b.sd)),
                    float(np.max(np.abs(a.mean - b.mean) / b.sd)))
    return worst


#: Relative tolerance of the quadrature convergence check.
QUAD_TOL = 0.01
#: Node counts tried in turn (the library default first; a forecast at 256
#: nodes costs ~4× one at 64 for a K = 3 pair-margin PMC, 512 was too slow).
G_LADDER = (64, 128, 256)


def choose_nodes(model: PMCModel, Y: np.ndarray, origins, H: int) -> tuple[int, float, float]:
    """Quadrature nodes for the forecasts and the gating of a fitted model.

    pmcprg integrates missing and future rows on a fixed grid of ``gap_nodes``
    nodes (default 64, ``pmcprg.pmc.gaps``). With strong copula dependence the
    conditional laws are narrower than the node spacing and the forecasts are
    silently wrong (README, "Library problems"). Returns (G, check of G,
    check of 64): the first G of G_LADDER whose forecasts change by less than
    QUAD_TOL when the nodes are doubled (``quad_diff``); if none does, the
    last G with the check of the step before it (flagged when > QUAD_TOL).
    Checks are 0 for the exact variants (no grid).
    """
    from pmcprg.pmc.gaps import needs_grid
    if not needs_grid(model):
        return 64, 0.0, 0.0
    first = None
    for G, G2 in zip(G_LADDER[:-1], G_LADDER[1:]):
        chk = quad_diff(model, Y, origins, H, G, G2)
        first = chk if first is None else first
        if chk <= QUAD_TOL:
            return G, chk, first
    return G_LADDER[-1], chk, first


def n_params(model: PMCModel) -> int:
    return rc.n_free_params(model)


def degenerate(model: PMCModel, Y: np.ndarray) -> bool:
    """real_series rule: a margin sd < 1 % of the series sd, or a state with π < 0.005."""
    return bool(rc.min_margin_sd(model) < rc.DEGENERATE_SD_RATIO * float(np.nanstd(Y))
                or np.min(model.stationary_pi) < rc.DEGENERATE_MIN_PI)


# ---------------------------------------------------------------------------
# Forecasts at the origins
# ---------------------------------------------------------------------------

def last_observed(prefix: np.ndarray) -> int:
    return int(np.nonzero(np.isfinite(prefix))[0][-1])


def pmc_rows(method: str, model: PMCModel, Yc: np.ndarray, case: dict, extra_fn=None,
             gap_nodes: int = 64) -> list[dict]:
    """pmcprg ``forecast`` at every origin, conditioned on Yc[:o] (NaN allowed)."""
    from pmcprg.pmc.gaps import needs_grid, reference_grid
    rows = []
    H, truth = case["H"], case["truth"]
    exact_mix = not model.variant.uses_copula and model.margin_structure != "pair"
    if exact_mix:
        mus = np.array([float(model.margin(k)._frozen.mean()) for k in range(model.K)])
        sds = np.array([float(model.margin(k)._frozen.std()) for k in range(model.K)])
    omega = reference_grid(model, gap_nodes).omega if needs_grid(model) else None
    for o in case["origins"]:
        fc = forecast(model, Yc[:o], H, quantiles=QLEVELS, gap_nodes=gap_nodes)
        fut = truth[o:o + H]
        crps = None
        if exact_mix and all(model.margin(k).dist_name == "norm" for k in range(model.K)):
            # State-margin mixture of Gaussians: exact CRPS (the quantile one, kept
            # in ``crps_q``, is checked against it).
            crps = crps_gauss_mixture(np.where(np.isfinite(fut), fut, 0.0), fc.state_probs, mus, sds)
        extra = dict(extra_fn(o)) if extra_fn else {}
        qcheck = np.zeros(H)
        if omega is not None:
            # Consistency of the returned quantiles with the forecast's own node law
            # (density × quadrature weight, the law of its mean and sd): distance of
            # the returned median to the node-law median, in predictive sd. pmcprg
            # returns quantiles far from that law for some multimodal predictive laws
            # (README, "Library problems", P3); > 0.25 marks the row.
            mass = fc.density * omega[None, :]
            cum = np.cumsum(mass, axis=1) / mass.sum(axis=1, keepdims=True)
            med = np.array([fc.nodes[min(int(np.searchsorted(c, 0.5)), len(fc.nodes) - 1)] for c in cum])
            qcheck = np.abs(fc.quantile_values[:, _QIDX[0.5]] - med) / np.maximum(fc.sd, 1e-12)
        out = rows_for(method, o, fut, fc.mean, fc.sd, fc.quantile_values, crps, extra)
        for r in out:
            r["q_check"] = float(qcheck[r["h"] - 1])
        rows += out
    return rows


def ar1_rows(method: str, par: dict, Yc: np.ndarray, case: dict, extra_fn=None) -> list[dict]:
    """Gaussian AR(1) forecast from the last observed row (a trailing gap skipped exactly)."""
    rows = []
    H, truth = case["H"], case["truth"]
    for o in case["origins"]:
        L = last_observed(Yc[:o])
        m, s = rc.ar1_forecast(float(Yc[L]), par, H + (o - 1 - L))
        m, s = m[-H:], s[-H:]
        fut = truth[o:o + H]
        rows += rows_for(method, o, fut, m, s, gauss_quantiles(m, s), crps_gauss(np.where(np.isfinite(fut), fut, 0), m, s),
                         extra_fn(o) if extra_fn else None)
    return rows


def persistence_rows(method: str, Yc: np.ndarray, case: dict, extra_fn=None) -> list[dict]:
    """Last observed value; predictive quantiles = last + empirical quantiles of the
    training changes y_{t+k} − y_t (k = h + trailing gap), both rows observed."""
    H, truth, n_train = case["H"], case["truth"], case["n_train"]
    ytr = Yc[:n_train]
    gaps = [o - 1 - last_observed(Yc[:o]) for o in case["origins"]]
    kmax = H + max(gaps)
    qd = np.empty((kmax + 1, len(QLEVELS)))
    for k in range(1, kmax + 1):
        d = ytr[k:] - ytr[:-k]
        d = d[np.isfinite(d)]
        qd[k] = np.quantile(d, QLEVELS)
    rows = []
    for o, g in zip(case["origins"], gaps):
        L = o - 1 - g
        last = float(Yc[L])
        qv = last + qd[g + 1: g + 1 + H]
        fut = truth[o:o + H]
        rows += rows_for(method, o, fut, np.full(H, last), None, qv, None, extra_fn(o) if extra_fn else None)
    return rows


def spike_extra(case: dict):
    """Row tags: a spike at the last row before the origin / among the last 3 rows."""
    sp = case["spikes"]

    def f(o):
        return {"spike_last": bool(sp[o - 1]), "spike_recent": bool(sp[max(0, o - 3):o].any())}
    return f


def gate(model: PMCModel, Y: np.ndarray, gap_nodes: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Sequential gating (flag_outliers, α = ALPHA): flagged rows → NaN.

    The gated filter is causal (row n tested with the filter of the rows before
    it, per-row threshold), so the flags of the whole series restricted to Y[:o]
    are those of Y[:o] — one call serves every origin.
    """
    f = flag_outliers(model, Y, alpha=ALPHA, sequential=True, gap_nodes=gap_nodes).flagged
    Yg = Y.copy()
    Yg[f] = np.nan
    return Yg, f


def mask_stats(mask: np.ndarray, case: dict, part: slice) -> dict:
    sp = case["spikes"][part]
    m = mask[part]
    obs = np.isfinite(case["Y"][part])
    clean = obs & ~sp
    return {"n_flag": int(m.sum()), "detect": float(m[sp].mean()) if sp.any() else np.nan,
            "false_per_1000": float(1000 * m[clean].mean()) if clean.any() else np.nan}


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

def _quiet():
    logging.getLogger("pmcprg").setLevel(logging.ERROR)


def task_select(spec: dict) -> dict:
    """Mote 20: fit one candidate (kind, K) on the clean training part; BIC."""
    _quiet()
    case = build_case("mote20__clean", spec["data_dir"], spec["quick"])
    Ytr = case["Y"][:case["n_train"]]
    t0 = time.perf_counter()
    model, trace = fit_kmeans(spec["kind"], spec["K"], Ytr)
    secs = time.perf_counter() - t0
    ll = float(classify(model, Ytr)[2])
    k = n_params(model)
    n_obs = int(np.isfinite(Ytr).sum())
    s = rc.model_summary(model)
    G, chk, chk64 = choose_nodes(model, case["Y"], case["origins"], case["H"])
    return {"select": [{"series": "mote20", "kind": spec["kind"], "K": spec["K"], "ll": ll, "n_params": k,
                        "bic": -2 * ll + k * math.log(n_obs), "degenerate": degenerate(model, Ytr),
                        "gap_nodes": G, "quad_check": chk, "quad_check_64": chk64,
                        "min_margin_sd": float(rc.min_margin_sd(model)), "min_pi": float(np.min(model.stationary_pi)),
                        "mean_sorted": s["mean_sorted"], "sd_sorted": s["sd_sorted"], "copulas": s["copulas"],
                        "seconds": secs, "n_iter": len(trace.log_liks)}],
            "models": {f"mote20__{family(spec['kind'], spec['K'])}": model.raw}}


def _evaluate(case: dict, fam: str, fit: str, model: PMCModel, rows: list, fits: list, *,
              trace=None, secs=np.nan, extra=None, plain: bool = True, gated: bool = True,
              plain_name: str | None = None, Y_cond: np.ndarray | None = None) -> None:
    """Record one fitted model and its forecasts.

    The quadrature nodes G are chosen per model (``choose_nodes``) and used by
    both the gating (``flag_outliers``) and the forecasts. ``plain``: forecasts
    conditioned on the series as observed (``Y_cond``, default the case's Y);
    ``gated``: on the series whose rows the model's own sequential gating
    flagged are set to NaN (``<fit>+gate``).
    """
    Y = case["Y"] if Y_cond is None else Y_cond
    n_train = case["n_train"]
    Ytr = case["Y"][:n_train]
    ex = spike_extra(case)
    G, chk, chk64 = choose_nodes(model, Y, case["origins"], case["H"])
    r = {"case": case["case"], "family": fam, "fit": fit, "seconds": secs,
         "n_iter": len(trace.log_liks) if trace is not None else np.nan,
         "gap_nodes": G, "quad_check": chk, "quad_check_64": chk64,
         "min_pi": float(np.min(model.stationary_pi)), "min_margin_sd": float(rc.min_margin_sd(model)),
         "degenerate": degenerate(model, Ytr)}
    s = rc.model_summary(model)
    r.update({k: s.get(k, "") for k in ("pi_sorted", "mean_sorted", "sd_sorted", "stay_sorted", "copulas")})
    if gated:
        Yg, f = gate(model, Y, G)
        r.update({f"gate_train_{k}": v for k, v in mask_stats(f, case, slice(0, n_train)).items()})
        r.update({f"gate_test_{k}": v for k, v in mask_stats(f, case, slice(n_train, len(Y))).items()})
        rows += pmc_rows(f"{fam}|{fit}+gate", model, Yg, case, ex, G)
    if plain:
        rows += pmc_rows(f"{fam}|{plain_name or fit}", model, Y, case, ex, G)
    if extra:
        r.update(extra)
    fits.append(r)


def _robust_extra(rf, case: dict, init_mask=None) -> dict:
    n_train, N = case["n_train"], len(case["Y"])
    pad = np.zeros(N - n_train, bool)
    out = {"n_fits": rf.n_fits, "converged": rf.converged,
           **{f"mask_{k}": v for k, v in mask_stats(np.r_[rf.mask, pad], case, slice(0, n_train)).items()}}
    if init_mask is not None:
        out.update({f"prescreen_{k}": v for k, v in
                    mask_stats(np.r_[init_mask, pad], case, slice(0, n_train)).items()})
    return out


def task_pmc(spec: dict) -> dict:
    """One case × one pmcprg family (real series): raw / robust / Hampel fits, gated or not.

    Fits (on Y[:n_train] of the case, k-means start, ICE "available" as in
    real_series; a pre-fitted model may be passed for the clean mote-20 case):

    * raw     — the fit on the series as observed;
    * robust  — ``robust_estimate`` (fit, flag, mask, refit from the k-means
                start of the unmasked rows, until a fixed point);
    * hampel  — the same with the Hampel pre-screen as ``initial_mask``
                (contaminated cases only).

    Forecasts: every fit conditioned on the observed Y[:o] (``raw``,
    ``robust``) or on the gated Y[:o] (``+gate``: rows flagged by the fit's own
    sequential gating set to NaN); the Hampel fit is only used gated.
    """
    _quiet()
    case = build_case(spec["case"], spec["data_dir"], spec["quick"])
    kind, K = spec["kind"], spec["K"]
    fam = family(kind, K)
    Ytr = case["Y"][:case["n_train"]]
    rows, fits = [], []
    t0 = time.perf_counter()
    if spec.get("model_raw") is not None:
        raw, trace = PMCModel.from_dict(spec["model_raw"]), None
    else:
        raw, trace = fit_kmeans(kind, K, Ytr)
    _evaluate(case, fam, "raw", raw, rows, fits, trace=trace, secs=time.perf_counter() - t0,
              extra={"ll_train": float(classify(raw, Ytr)[2])})
    variants = [("robust", None)]
    if spec.get("hampel", False):
        variants.append(("hampel", hampel_nan(Ytr)))
    for name, init_mask in variants:
        t0 = time.perf_counter()
        rf = robust_estimate(rc.initial_model(kind, K, Ytr), Ytr, kmeans_cfg(kind),
                             flag_cfg={"alpha": ALPHA}, initial_mask=init_mask)
        _evaluate(case, fam, name, rf.model, rows, fits, trace=rf.trace, secs=time.perf_counter() - t0,
                  extra=_robust_extra(rf, case, init_mask), plain=(name == "robust"))
    df = pd.DataFrame(rows)
    df.insert(0, "case", case["case"])
    return {"records": df, "fits": fits}


def task_baselines(spec: dict) -> dict:
    """AR(1) (raw, gated, flag-and-mask + gated; exact Gaussian gate) and persistence."""
    _quiet()
    case = build_case(spec["case"], spec["data_dir"], spec["quick"])
    Y, n_train = case["Y"], case["n_train"]
    Ytr = Y[:n_train]
    ex = spike_extra(case)
    rows, fits = [], []
    t0 = time.perf_counter()
    par = rc.ar1_fit(Ytr)
    secs = time.perf_counter() - t0
    f = ar1_flags(Y, par)
    Yg = np.where(f, np.nan, Y)
    fits.append({"case": case["case"], "family": "ar1", "fit": "raw", "seconds": secs, **par,
                 **{f"gate_train_{k}": v for k, v in mask_stats(f, case, slice(0, n_train)).items()},
                 **{f"gate_test_{k}": v for k, v in mask_stats(f, case, slice(n_train, len(Y))).items()}})
    rows += ar1_rows("ar1|raw", par, Y, case, ex)
    rows += ar1_rows("ar1|raw+gate", par, Yg, case, ex)
    t0 = time.perf_counter()
    rpar, mask, nf, conv = ar1_robust(Ytr)
    f = ar1_flags(Y, rpar)
    Yg = np.where(f, np.nan, Y)
    fits.append({"case": case["case"], "family": "ar1", "fit": "robust", "seconds": time.perf_counter() - t0,
                 "n_fits": nf, "converged": conv, **rpar,
                 **{f"mask_{k}": v for k, v in mask_stats(np.r_[mask, np.zeros(len(Y) - n_train, bool)], case,
                                                          slice(0, n_train)).items()},
                 **{f"gate_test_{k}": v for k, v in mask_stats(f, case, slice(n_train, len(Y))).items()}})
    rows += ar1_rows("ar1|robust+gate", rpar, Yg, case, ex)
    rows += persistence_rows("persistence|raw", Y, case, ex)
    df = pd.DataFrame(rows)
    df.insert(0, "case", case["case"])
    return {"records": df, "fits": fits}


def task_fixture(spec: dict) -> dict:
    """One simulated series of report/erroneous_data: every pmcprg method + baselines.

    Fits start at the true model (as in report/erroneous_data); ``clean`` is the
    fit on the series before contamination, conditioned on the clean series;
    ``true`` the true model, conditioned on the contaminated series.
    """
    _quiet()
    case = build_case(spec["case"], spec["data_dir"], spec["quick"])
    name = case["fixture"]
    kind, K = FIXTURES[name]
    fam = family(kind, K)
    model = PMCModel(ROOT / "pmcprg" / "pmc" / "models" / f"{name}.toml")
    cfg = fixture_cfg(model)
    Y, Y0, n_train = case["Y"], case["Y_clean"], case["n_train"]
    Ytr = Y[:n_train]
    ex = spike_extra(case)
    rows, fits = [], []
    _evaluate(case, fam, "true", model, rows, fits)
    t0 = time.perf_counter()
    cl, trc = ice(model, Y0[:n_train], cfg)
    _evaluate(case, fam, "clean", cl, rows, fits, trace=trc, secs=time.perf_counter() - t0,
              gated=False, Y_cond=Y0)
    t0 = time.perf_counter()
    raw, tr = ice(model, Ytr, cfg)
    _evaluate(case, fam, "raw", raw, rows, fits, trace=tr, secs=time.perf_counter() - t0)
    for nm, init_mask in (("robust", None), ("hampel", hampel_nan(Ytr))):
        t0 = time.perf_counter()
        rf = robust_estimate(model, Ytr, cfg, flag_cfg={"alpha": ALPHA}, initial_mask=init_mask)
        _evaluate(case, fam, nm, rf.model, rows, fits, trace=rf.trace, secs=time.perf_counter() - t0,
                  extra=_robust_extra(rf, case, init_mask), plain=(nm == "robust"))
    out = task_baselines(spec)
    # AR(1) and persistence on the clean series (references)
    par0 = rc.ar1_fit(Y0[:n_train])
    clean_case = dict(case, Y=Y0)
    extra_rows = (ar1_rows("ar1|clean", par0, Y0, clean_case, ex)
                  + persistence_rows("persistence|clean", Y0, clean_case, ex))
    df = pd.concat([pd.DataFrame(rows), out["records"].drop(columns="case"), pd.DataFrame(extra_rows)],
                   ignore_index=True)
    df.insert(0, "case", case["case"])
    return {"records": df, "fits": fits + out["fits"]}


def pmm_rows(case: dict, pmm_fc: pd.DataFrame) -> pd.DataFrame:
    """Score pmmforecast's Gaussian predictive laws (mean, sd per origin and h).

    ``pmm|raw``: the higher-likelihood fit; ``pmm_default|raw``: the fit from
    pmmforecast's default starts (``pmm_side.fit_pmm``).
    """
    ex = spike_extra(case)
    H, truth = case["H"], case["truth"]
    rows = []
    for (fit, o), g in pmm_fc.groupby(["fit", "origin"], sort=True):
        method = "pmm|raw" if fit == "best" else "pmm_default|raw"
        g = g.sort_values("h")
        m, s = g["mean"].to_numpy(), g["sd"].to_numpy()
        fut = truth[int(o):int(o) + H]
        rows += rows_for(method, int(o), fut, m, s, gauss_quantiles(m, s),
                         crps_gauss(np.where(np.isfinite(fut), fut, 0), m, s), ex(int(o)))
    df = pd.DataFrame(rows)
    df.insert(0, "case", case["case"])
    return df


TASKS = {"select": task_select, "pmc": task_pmc, "baselines": task_baselines, "fixture": task_fixture}


def run_task(spec: dict):
    t0 = time.perf_counter()
    try:
        res = TASKS[spec["task"]](spec)
    except Exception as exc:  # recorded; the campaign goes on
        import traceback
        res = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-2000:]}
    return spec, res, time.perf_counter() - t0
