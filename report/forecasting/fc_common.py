"""
fc_common.py — data, models, scores and tasks of the forecasting study (forecasting).

pmcprg side only (run with the awesomePMC interpreter); everything that imports
pmmforecast lives in ``pmm_side.py``. Data preparation, split, fits and baselines
are those of ``report/real_series`` (``rs_common``), so that the Aotizhongxin
numbers are comparable with its section 4; the contamination and the Hampel
pre-screen are those of ``report/erroneous_data``.

Every task is a pure function of its spec (seeds fixed in the spec), so the
results do not depend on the number of worker processes.

pmcprg's WARNINGs are not printed by the tasks; they are collected with the
step that raised them (``WarningLog``) and returned with the task's results,
in particular the quadrature WARNING of ``pmcprg.pmc.gaps`` (``quad_error``
above its limit) and the fallback of the missing-value quantiles.
"""

from __future__ import annotations

import logging
import math
import re
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

from pmcprg.pmc import (  # noqa: E402
    PMCModel,
    classify,
    flag_outliers,
    forecast,
    gap_posterior,
    ice,
    robust_estimate,
    simulate,
)

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
#: Seeds of the contaminated Aotizhongxin cases on which the copula PMC is fitted
#: (every model with an exact filter runs on all REAL_SEEDS).
REAL_PMC_SEEDS = 1
#: False: the copula PMC fitted on contaminated Aotizhongxin is only fitted and
#: its quadrature checked, not forecast (a forecast at G = 256 costs 9 s, about
#: 70 min per case and seed).
CONTAMINATED_PMC_FORECAST = True


# ---------------------------------------------------------------------------
# pmcprg WARNINGs, collected per task step
# ---------------------------------------------------------------------------

_QUAD_RE = re.compile(r"relative error ([0-9.eE+-]+) > ([0-9.eE+-]+) .*\((\d+) missing rows, gap_nodes = (\d+)\)")


class WarningLog(logging.Handler):
    """Collects the WARNINGs of the ``pmcprg`` loggers instead of printing them.

    Each record is counted under (step, kind, template): ``step`` is the
    current step of the task (``set_step``), ``kind`` is ``quad_error`` (the
    WARNING of ``pmcprg.pmc.gaps`` when ``GapPosterior.quad_error`` exceeds
    its limit; its value, limit, missing rows and G are parsed),
    ``quantile_fallback`` (the monotone-CDF fallback of the missing-value
    quantiles) or ``other``.
    """

    def __init__(self):
        super().__init__(logging.WARNING)
        self.step = ""
        self.items: dict = {}

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        kind = ("quad_error" if msg.startswith("Missing-data quadrature not converged") else
                "quantile_fallback" if msg.startswith("Quantiles of the missing values") else "other")
        key = (self.step, record.name, kind, str(record.msg)[:160])
        it = self.items.setdefault(key, {"step": self.step, "logger": record.name, "kind": kind,
                                         "template": str(record.msg)[:160], "count": 0, "example": msg[:300],
                                         "max_value": np.nan, "limit": np.nan, "n_missing": np.nan,
                                         "gap_nodes": np.nan})
        it["count"] += 1
        m = _QUAD_RE.search(msg) if kind == "quad_error" else None
        if m and not (float(m.group(1)) <= it["max_value"]):
            it.update(max_value=float(m.group(1)), limit=float(m.group(2)), n_missing=int(m.group(3)),
                      gap_nodes=int(m.group(4)), example=msg[:300])

    def take(self) -> list[dict]:
        out = list(self.items.values())
        self.items = {}
        self.step = ""
        return out


WARNINGS = WarningLog()


def set_step(label: str) -> None:
    WARNINGS.step = label


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


#: Quantile levels requested from the grid (copula) models: the coverage levels.
#: Their CRPS and PIT come from the node law (``node_law_scores``); the 200-level
#: quantile CRPS costs ~30× more with pmcprg's quantile code (README, "Cost")
#: and is computed at one origin per model for comparison.
QCOV = (0.025, 0.10, 0.25, 0.5, 0.75, 0.90, 0.975)


def _seg_sq(L, a, b):
    """∫ over a segment of length L of g², g linear from a to b."""
    return L * (a * a + a * b + b * b) / 3.0


def node_law_scores(y: np.ndarray, nodes: np.ndarray, mass: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CRPS and PIT of y under each row's node law, exactly for its piecewise-linear CDF.

    Row k of (nodes, mass) is a discrete law (``Forecast.grid_nodes`` /
    ``grid_mass``: the quadrature nodes of horizon k and their predictive
    masses). Each node's mass is spread uniformly on its cell (between the
    mid-points of the sorted nodes; half a spacing beyond the end nodes), so
    the CDF F is piecewise linear, and CRPS = ∫ (F(x) − 1{x ≥ y})² dx is
    integrated in closed form segment by segment. NaN y gives NaN.
    """
    H = nodes.shape[0]
    crps, pit = np.full(H, np.nan), np.full(H, np.nan)
    for k in range(H):
        if not np.isfinite(y[k]):
            continue
        o = np.argsort(nodes[k])
        x, m = nodes[k][o], np.clip(mass[k][o], 0.0, None)
        m = m / m.sum()
        b = np.empty(x.size + 1)
        b[1:-1] = 0.5 * (x[1:] + x[:-1])
        b[0], b[-1] = x[0] - 0.5 * (x[1] - x[0]), x[-1] + 0.5 * (x[-1] - x[-2])
        F = np.r_[0.0, np.cumsum(m)]
        F[-1] = 1.0
        L = np.diff(b)
        yk = float(y[k])
        if yk <= b[0]:
            crps[k], pit[k] = (b[0] - yk) + _seg_sq(L, 1 - F[:-1], 1 - F[1:]).sum(), 0.0
            continue
        if yk >= b[-1]:
            crps[k], pit[k] = _seg_sq(L, F[:-1], F[1:]).sum() + (yk - b[-1]), 1.0
            continue
        j = int(np.searchsorted(b, yk, side="right")) - 1        # b[j] <= y < b[j+1]
        Fy = F[j] + (F[j + 1] - F[j]) * ((yk - b[j]) / L[j] if L[j] > 0 else 0.0)
        left = _seg_sq(L[:j], F[:j], F[1:j + 1]).sum() + _seg_sq(yk - b[j], F[j], Fy)
        right = _seg_sq(b[j + 1] - yk, 1 - Fy, 1 - F[j + 1]) + _seg_sq(L[j + 1:], 1 - F[j + 1:-1], 1 - F[j + 2:]).sum()
        crps[k], pit[k] = left + right, Fy
    return crps, pit


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
    ``run_forecasting.py --quick``). The closed form was introduced when
    pmcprg's gap quadrature collapsed for φ ≳ 0.998 (README, P1, fixed since);
    it is kept because it is exact and costs nothing.
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


def node_law_quantiles(nodes: np.ndarray, mass: np.ndarray, levels) -> np.ndarray:
    """Quantiles of each row's node law, inverse of its piecewise-linear CDF (``node_law_scores``)."""
    out = np.empty((nodes.shape[0], len(levels)))
    for k in range(nodes.shape[0]):
        o = np.argsort(nodes[k])
        x, m = nodes[k][o], np.clip(mass[k][o], 0.0, None)
        b = np.r_[x[0] - 0.5 * (x[1] - x[0]), 0.5 * (x[1:] + x[:-1]), x[-1] + 0.5 * (x[-1] - x[-2])]
        F = np.r_[0.0, np.cumsum(m / m.sum())]
        out[k] = np.interp(levels, F, b)
    return out


def quad_diff(model: PMCModel, Y: np.ndarray, origins, H: int, G: int, G_ref: int) -> tuple[float, float]:
    """Convergence of the forecasts in the quadrature, at the first, middle and last origin.

    Returns (mean/sd change, tail change): the largest change of the predictive
    mean and sd between G and G_ref nodes, in units of the G_ref sd (inf when a
    predictive sd collapses, < 1e-3 of the series sd); and the largest change of
    the 2.5 / 97.5 % quantiles of the node law (the law the scores use,
    ``node_law_scores``), in the same unit.
    """
    sd_y = float(np.nanstd(Y))
    worst, tail = 0.0, 0.0
    origins = list(origins)
    for o in sorted({origins[0], origins[len(origins) // 2], origins[-1]}):
        a = forecast(model, Y[:o], H, quantiles=(0.5,), gap_nodes=G)
        b = forecast(model, Y[:o], H, quantiles=(0.5,), gap_nodes=G_ref)
        if np.min(a.sd) < 1e-3 * sd_y or np.min(b.sd) < 1e-3 * sd_y:
            return math.inf, math.inf
        worst = max(worst, float(np.max(np.abs(a.sd - b.sd) / b.sd)),
                    float(np.max(np.abs(a.mean - b.mean) / b.sd)))
        qa = node_law_quantiles(np.asarray(a.grid_nodes), np.asarray(a.grid_mass), (0.025, 0.975))
        qb = node_law_quantiles(np.asarray(b.grid_nodes), np.asarray(b.grid_mass), (0.025, 0.975))
        tail = max(tail, float(np.max(np.abs(qa - qb) / b.sd[:, None])))
    return worst, tail


def quad_ratio(model: PMCModel, Y: np.ndarray, origins, H: int, G: int) -> float:
    """Largest quad_error / WARNING limit of the forecast chains at the check origins of ``quad_diff``."""
    origins = list(origins)
    ratio = 0.0
    for o in sorted({origins[0], origins[len(origins) // 2], origins[-1]}):
        qe, lim = quad_error_of(model, np.r_[Y[:o], np.full(H, np.nan)], G)
        ratio = max(ratio, qe / lim)
    return ratio


#: Relative tolerance of the quadrature convergence check.
QUAD_TOL = 0.01
#: Tolerance on the change of the node-law 2.5 / 97.5 % quantiles (the CRPS and
#: the coverage are scored on the node law).
QUAD_TOL_TAIL = 0.05
#: Node counts tried in turn (the library default first; a forecast at 256
#: nodes costs ~4× one at 64 for a K = 3 pair-margin PMC, 512 was too slow).
G_LADDER = (64, 128, 256)


def choose_nodes(model: PMCModel, Y: np.ndarray, origins, H: int) -> dict:
    """Quadrature nodes for the forecasts and the gating of a fitted model.

    pmcprg integrates missing and future rows on grids of ``gap_nodes`` nodes
    (default 64, ``pmcprg.pmc.gaps``). Before the local grids of pmcprg's P1
    fix, strong copula dependence made the conditional laws narrower than the
    node spacing and the forecasts were silently wrong (README, P1); this check
    is kept as an independent convergence test. ``gap_nodes``: the first G of
    G_LADDER whose predictive means and sds change by less than QUAD_TOL, and
    whose node-law 2.5 / 97.5 % quantiles by less than QUAD_TOL_TAIL, of the
    predictive sd when G is doubled (``quad_diff``), else the last G (flagged:
    ``quad_check`` > QUAD_TOL or ``quad_check_tail`` > QUAD_TOL_TAIL). Also
    recorded: the check at G = 64 (``quad_check_64``) and pmcprg's own diagnostic, quad_error / WARNING
    limit of the forecast chains (``quad_ratio``, at G, and at 64). The ratio
    is not required: it is a screen, not a bound (pmcprg's CHANGELOG), and on
    the §4 Aotizhongxin model the forecasts at G = 64 are within 3e-4
    predictive sd of G = 512 while some forecast chains exceed the limit.
    Checks are 0 for the exact variants (no grid).
    """
    from pmcprg.pmc.gaps import needs_grid
    if not needs_grid(model):
        return {"gap_nodes": 64, "quad_check": 0.0, "quad_check_64": 0.0, "quad_check_tail": 0.0,
                "quad_ratio": 0.0, "quad_ratio_64": 0.0}
    out = {}
    for G, G2 in zip(G_LADDER[:-1], G_LADDER[1:]):
        chk, tail = quad_diff(model, Y, origins, H, G, G2)
        if G == G_LADDER[0]:
            out["quad_check_64"] = chk
            out["quad_ratio_64"] = quad_ratio(model, Y, origins, H, G)
        if chk <= QUAD_TOL and tail <= QUAD_TOL_TAIL:
            break
    else:
        G = G_LADDER[-1]
    out.update(gap_nodes=G, quad_check=chk, quad_check_tail=tail,
               quad_ratio=out["quad_ratio_64"] if G == G_LADDER[0] else quad_ratio(model, Y, origins, H, G))
    return out


def quad_error_of(model: PMCModel, Y: np.ndarray, G: int) -> tuple[float, float]:
    """(``GapPosterior.quad_error``, its WARNING limit) of a conditioning series.

    NaN for the exact variants or a series without missing rows (no grid).
    The limit is ``gaps.quad_warn_limit``: QUAD_WARN · √(number of runs of
    missing rows).
    """
    from pmcprg.pmc.gaps import needs_grid, quad_warn_limit
    miss = ~np.isfinite(Y)
    if not needs_grid(model) or not miss.any():
        return np.nan, np.nan
    post = gap_posterior(model, Y, gap_nodes=G, xi=False)
    return float(post.quad_error), float(quad_warn_limit(miss))


def _node_median(nodes: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """Median of each row's discrete law (nodes, masses; nodes in any order)."""
    o = np.argsort(nodes, axis=1)
    n_s, m_s = np.take_along_axis(nodes, o, 1), np.take_along_axis(mass, o, 1)
    cum = np.cumsum(m_s, axis=1) / m_s.sum(axis=1, keepdims=True)
    idx = np.minimum((cum < 0.5).sum(axis=1), nodes.shape[1] - 1)
    return n_s[np.arange(len(n_s)), idx]


def node_median_check(fc, omega: np.ndarray, q50: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Consistency of the returned median with the forecast's own discrete law.

    ``q_check``: distance, in predictive sd, of the returned median to the
    median of the node law of each horizon, on that horizon's own quadrature
    nodes and masses (``Forecast.grid_nodes`` / ``grid_mass``: the reference
    grid or a local one). ``q_check_ref``: the same on the reference nodes,
    density × quadrature weight ``omega`` (the check of the stopped run; with
    local grids the reference nodes may not resolve a narrow law). pmcprg's
    P3 returned quantiles far from that law; > 0.25 marks the row. ``q50``:
    the returned medians.
    """
    sd = np.maximum(fc.sd, 1e-12)
    own = np.abs(q50 - _node_median(np.asarray(fc.grid_nodes), np.asarray(fc.grid_mass))) / sd
    dens = np.asarray(fc.density)
    ref = np.abs(q50 - _node_median(np.broadcast_to(fc.nodes, dens.shape), dens * omega[None, :])) / sd
    return own, ref


def leading_gap(Y: np.ndarray) -> int:
    """Number of missing rows at the start of Y (pmcprg's leading-gap problem, README)."""
    fin = np.isfinite(np.asarray(Y, float))
    return int(np.argmax(fin)) if fin.any() else len(fin)


def max_diag_tau(model: PMCModel) -> float:
    """Largest Kendall τ of the diagonal copulas c_ii (NaN without copulas)."""
    taus = [float(c["tau"]) for c in model.raw.get("copulas", []) if c.get("i") == c.get("j") and "tau" in c]
    return max(taus) if taus else np.nan


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
    grid = needs_grid(model)
    omega = reference_grid(model, gap_nodes).omega if grid else None
    origins = list(case["origins"])
    o_check = origins[len(origins) // 2]
    for o in origins:
        fut = truth[o:o + H]
        extra = dict(extra_fn(o)) if extra_fn else {}
        if not grid:
            fc = forecast(model, Yc[:o], H, quantiles=QLEVELS, gap_nodes=gap_nodes)
            crps = None
            if exact_mix and all(model.margin(k).dist_name == "norm" for k in range(model.K)):
                # State-margin mixture of Gaussians: exact CRPS (the quantile one, kept
                # in ``crps_q``, is checked against it).
                crps = crps_gauss_mixture(np.where(np.isfinite(fut), fut, 0.0), fc.state_probs, mus, sds)
            rows += rows_for(method, o, fut, fc.mean, fc.sd, fc.quantile_values, crps, extra)
            continue
        # Grid (copula) models: the coverage levels only; CRPS and PIT of the node law
        # (exact for its piecewise-linear CDF, ``node_law_scores``); at the middle origin
        # the 200-level quantile CRPS too (``crps_q``), for comparison.
        qs = QLEVELS if o == o_check else QCOV
        fc = forecast(model, Yc[:o], H, quantiles=qs, gap_nodes=gap_nodes)
        qv = np.full((H, len(QLEVELS)), np.nan)
        for j, q in enumerate(qs):
            qv[:, _QIDX[q]] = fc.quantile_values[:, j]
        gn, gm = np.asarray(fc.grid_nodes), np.asarray(fc.grid_mass)
        crps, pit = node_law_scores(fut, gn, gm)
        # Consistency of the returned quantiles with the forecast's own node law
        # (README, P3 — fixed in pmcprg; kept as a per-row check): the median
        # (q_check) and the 2.5 / 97.5 % quantiles (q_check_tail), in predictive sd.
        qcheck, qcheck_ref = node_median_check(fc, omega, qv[:, _QIDX[0.5]])
        qn = node_law_quantiles(gn, gm, (0.025, 0.975))
        qtail = np.max(np.abs(qv[:, [_QIDX[0.025], _QIDX[0.975]]] - qn), axis=1) / np.maximum(fc.sd, 1e-12)
        out = rows_for(method, o, fut, fc.mean, fc.sd, qv, crps, extra)
        for r in out:
            k = r["h"] - 1
            # Scores of the node law: PIT, coverage and 95 % width (the returned
            # quantiles are only checked, q_check / q_check_tail).
            r["pit"] = float(pit[k])
            for lev, (lo, hi) in COVERAGE.items():
                r[f"c{lev}"] = bool(lo <= pit[k] <= hi)
            r["w95"] = float(qn[k, 1] - qn[k, 0])
            if o != o_check:
                r["crps_q"] = np.nan
            r["q_check"] = float(qcheck[k])
            r["q_check_ref"] = float(qcheck_ref[k])
            r["q_check_tail"] = float(qtail[k])
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
    """pmcprg WARNINGs go to ``WARNINGS`` (collected, not printed)."""
    lg = logging.getLogger("pmcprg")
    lg.setLevel(logging.WARNING)
    lg.propagate = False
    if WARNINGS not in lg.handlers:
        lg.addHandler(WARNINGS)


def task_select(spec: dict) -> dict:
    """Mote 20: fit one candidate (kind, K) on the clean training part; BIC."""
    _quiet()
    case = build_case("mote20__clean", spec["data_dir"], spec["quick"])
    Ytr = case["Y"][:case["n_train"]]
    fam = family(spec["kind"], spec["K"])
    set_step(f"{fam}|select fit")
    t0 = time.perf_counter()
    model, trace = fit_kmeans(spec["kind"], spec["K"], Ytr)
    secs = time.perf_counter() - t0
    set_step(f"{fam}|select classify")
    ll = float(classify(model, Ytr)[2])
    k = n_params(model)
    n_obs = int(np.isfinite(Ytr).sum())
    s = rc.model_summary(model)
    set_step(f"{fam}|select choose_nodes")
    qd = choose_nodes(model, case["Y"], case["origins"], case["H"])
    return {"select": [{"series": "mote20", "kind": spec["kind"], "K": spec["K"], "ll": ll, "n_params": k,
                        "bic": -2 * ll + k * math.log(n_obs), "degenerate": degenerate(model, Ytr),
                        **qd, "lead_gap": leading_gap(Ytr), "max_diag_tau": max_diag_tau(model),
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
    set_step(f"{fam}|{fit} choose_nodes")
    qd = choose_nodes(model, Y, case["origins"], case["H"])
    G = qd["gap_nodes"]
    r = {"case": case["case"], "family": fam, "fit": fit, "seconds": secs,
         "n_iter": len(trace.log_liks) if trace is not None else np.nan, **qd,
         "max_diag_tau": max_diag_tau(model), "lead_gap_fit": leading_gap(Ytr), "lead_gap_cond": leading_gap(Y),
         "min_pi": float(np.min(model.stationary_pi)), "min_margin_sd": float(rc.min_margin_sd(model)),
         "degenerate": degenerate(model, Ytr)}
    s = rc.model_summary(model)
    r.update({k: s.get(k, "") for k in ("pi_sorted", "mean_sorted", "sd_sorted", "stay_sorted", "copulas")})
    if plain:
        set_step(f"{fam}|{fit} quad_error observed")
        r["quad_error_obs"], r["quad_limit_obs"] = quad_error_of(model, Y, G)
    if gated:
        set_step(f"{fam}|{fit} gate")
        Yg, f = gate(model, Y, G)
        r.update({f"gate_train_{k}": v for k, v in mask_stats(f, case, slice(0, n_train)).items()})
        r.update({f"gate_test_{k}": v for k, v in mask_stats(f, case, slice(n_train, len(Y))).items()})
        r["lead_gap_gated"] = leading_gap(Yg)
        set_step(f"{fam}|{fit} quad_error gated")
        r["quad_error_gated"], r["quad_limit_gated"] = quad_error_of(model, Yg, G)
        set_step(f"{fam}|{fit}+gate forecast")
        rows += pmc_rows(f"{fam}|{fit}+gate", model, Yg, case, ex, G)
    if plain:
        set_step(f"{fam}|{plain_name or fit} forecast")
        rows += pmc_rows(f"{fam}|{plain_name or fit}", model, Y, case, ex, G)
    if extra:
        r.update(extra)
    fits.append(r)


def _robust_extra(rf, case: dict, init_mask=None) -> dict:
    n_train, N = case["n_train"], len(case["Y"])
    pad = np.zeros(N - n_train, bool)
    Ym = np.where(rf.mask, np.nan, case["Y"][:n_train])
    out = {"n_fits": rf.n_fits, "converged": rf.converged, "lead_gap_fit": leading_gap(Ym),
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
    fcst = spec.get("forecast", True)   # False: fits and quadrature checks only (P4)
    t0 = time.perf_counter()
    set_step(f"{fam}|raw fit")
    if spec.get("model_raw") is not None:
        raw, trace = PMCModel.from_dict(spec["model_raw"]), None
    else:
        raw, trace = fit_kmeans(kind, K, Ytr)
    _evaluate(case, fam, "raw", raw, rows, fits, trace=trace, secs=time.perf_counter() - t0,
              extra={"ll_train": float(classify(raw, Ytr)[2])}, plain=fcst, gated=fcst)
    variants = [("robust", None)]
    if spec.get("hampel", False):
        variants.append(("hampel", hampel_nan(Ytr)))
    for name, init_mask in variants:
        t0 = time.perf_counter()
        set_step(f"{fam}|{name} fit")
        rf = robust_estimate(rc.initial_model(kind, K, Ytr), Ytr, kmeans_cfg(kind),
                             flag_cfg={"alpha": ALPHA}, initial_mask=init_mask)
        _evaluate(case, fam, name, rf.model, rows, fits, trace=rf.trace, secs=time.perf_counter() - t0,
                  extra=_robust_extra(rf, case, init_mask), plain=fcst and (name == "robust"), gated=fcst)
    df = pd.DataFrame(rows)
    df.insert(0, "case", case["case"])
    for f in fits:
        f["forecast"] = fcst
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
    set_step(f"{fam}|clean fit")
    cl, trc = ice(model, Y0[:n_train], cfg)
    _evaluate(case, fam, "clean", cl, rows, fits, trace=trc, secs=time.perf_counter() - t0,
              gated=False, Y_cond=Y0)
    t0 = time.perf_counter()
    set_step(f"{fam}|raw fit")
    raw, tr = ice(model, Ytr, cfg)
    _evaluate(case, fam, "raw", raw, rows, fits, trace=tr, secs=time.perf_counter() - t0)
    for nm, init_mask in (("robust", None), ("hampel", hampel_nan(Ytr))):
        t0 = time.perf_counter()
        set_step(f"{fam}|{nm} fit")
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
    _quiet()
    WARNINGS.take()
    try:
        res = TASKS[spec["task"]](spec)
    except Exception as exc:  # recorded; the campaign goes on
        import traceback
        res = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-2000:]}
    res["warnings"] = [{"task": spec["task"], "case": spec.get("case", ""), **w} for w in WARNINGS.take()]
    return spec, res, time.perf_counter() - t0
