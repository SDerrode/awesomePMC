"""
rs_common.py — data preparation, models, baselines and tasks of the real-series
study (A4): unsupervised PMC/HMC estimation, imputation and classification on
three real series with gaps (tsNH4, Beijing PM2.5, PAMAP2).

Every task is a pure function of its spec dict (seeds are fixed in the spec),
so the results do not depend on the number of worker processes. The data are
read from the data folder given on the command line and never copied into the
repository; large outputs (fitted models, label arrays) go to ``--out``.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp

from pmcprg.missing import metrics, patterns
from pmcprg.pmc import PMCModel, classify, forecast, ice, impute, sem
from pmcprg.pmc._estim_common import build_multistart_inits, m_step, warmstart_from_kmeans

DEFAULT_DATA = "/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/data/timeseries"

#: Copula families selected at every M-step (1-parameter, no Product).
COPULA_CANDIDATES = ["Gauss", "Clayton", "GH", "Frank"]
#: GICE margin families (A23 §5 set).
GICE_CANDIDATES = ["norm", "gamma", "invgamma", "betaprime"]
#: Family rule of the GICE margins: BIC, so that 4-parameter families must earn
#: their extra parameters (the library default "mle" always favours them).
GICE_RULE = "bic"
#: Relative jitter of the extra starts (library multistart perturbation).
START_JITTER = 0.25
#: Base seed of the multistart perturbations (same starting points for every
#: strategy of a (series, K, model) cell: paired comparison).
START_SEED = 2026
ICE_MAX_ITER = 50
ICE_TOL = 1e-4
SEM_MAX_ITER = 30
IMPUTE_DRAWS = 200
QUANTILES = (0.05, 0.5, 0.95)

MODEL_KINDS = ("hmc_in", "pmc_state", "pmc_pair", "pmc_state_gice")
STRATEGIES = ("available", "impute", "sem", "complete")

#: PAMAP2 activity groups. Transient windows (activity 0) get -1 and are
#: excluded from every error rate.
PAMAP2_ACTIVITIES = {
    1: "lying", 2: "sitting", 3: "standing", 4: "walking", 5: "running",
    6: "cycling", 7: "Nordic walking", 12: "ascending stairs",
    13: "descending stairs", 16: "vacuum cleaning", 17: "ironing",
    24: "rope jumping",
}
GROUPS_K2 = {1: 0, 2: 0, 3: 0, 17: 0, 4: 1, 5: 1, 6: 1, 7: 1, 12: 1, 13: 1, 16: 1, 24: 1}
GROUP_NAMES_K2 = ("rest", "active")
GROUPS_K3 = {1: 0, 2: 0, 3: 0, 17: 0, 4: 1, 6: 1, 12: 1, 13: 1, 16: 1, 5: 2, 7: 2, 24: 2}
GROUP_NAMES_K3 = ("rest", "locomotion", "vigorous")
PAMAP2_WINDOW = 50          # samples per window: 0.5 s at 100 Hz → 2 Hz series
PAMAP2_MIN_MISSING = 5      # window missing when ≥ 5 of its 50 samples are (≥ 10 %)
PAMAP2_MCAR_RATE = 0.10     # added MCAR blocks (fraction of the series)
PAMAP2_MCAR_BLOCK = 20      # windows per block (10 s)
DEQUANT_SEED = 11           # dequantisation jitter of the integer PM2.5 values
ORACLE_PRIOR_FLOOR = 1e-4   # added to the supervised prior before renormalisation
#: A fit is flagged degenerate when one of its margins has a standard deviation
#: below this fraction of the standard deviation of the observed series.
DEGENERATE_SD_RATIO = 0.01
#: ... or when a state has a stationary probability below this value.
DEGENERATE_MIN_PI = 0.005

logger = logging.getLogger("real_series")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def gap_lengths(miss: np.ndarray) -> np.ndarray:
    """Length of the run of missing values containing each position (0 if observed)."""
    miss = np.asarray(miss, bool)
    out = np.zeros(miss.size, dtype=int)
    n = miss.size
    i = 0
    while i < n:
        if miss[i]:
            j = i
            while j < n and miss[j]:
                j += 1
            out[i:j] = j - i
            i = j
        else:
            i += 1
    return out


def gap_class(length: np.ndarray) -> np.ndarray:
    """'1-2', '3-20' or '>20' for the lengths of :func:`gap_lengths` (> 0)."""
    return np.where(length <= 2, "1-2", np.where(length <= 20, "3-20", ">20"))


def align_labels(x_true: np.ndarray, x_hat: np.ndarray, K: int) -> np.ndarray:
    """Relabel ``x_hat`` by the permutation minimising the error (Hungarian)."""
    C = np.zeros((K, K), dtype=int)
    np.add.at(C, (x_true, x_hat), 1)
    rows, cols = linear_sum_assignment(-C)
    relabel = np.arange(K)
    relabel[cols] = rows
    return relabel[x_hat]


def agreement(a: np.ndarray, b: np.ndarray, K: int, sel: np.ndarray | None = None) -> float:
    """Share of positions with the same label after the best permutation."""
    if sel is not None:
        a, b = a[sel], b[sel]
    return float(np.mean(align_labels(a, b, K) == a))


def model_summary(model: PMCModel) -> dict:
    """Compact, sortable description of a fitted model (states by margin mean)."""
    K = model.K
    pi = np.asarray(model.stationary_pi, float)
    if model.margin_structure == "state":
        means = np.array([model.margin(i)._frozen.mean() for i in range(K)])
        sds = np.array([model.margin(i)._frozen.std() for i in range(K)])
    else:
        means = np.array([model.margin(i, i)._frozen.mean() for i in range(K)])
        sds = np.array([model.margin(i, i)._frozen.std() for i in range(K)])
    order = np.argsort(means)
    p = (np.asarray(model.prior_p, float) if not model.variant.has_markov_prior
         else pi[:, None] * np.asarray(model.transition_A, float))
    stay = np.diag(p) / np.maximum(pi, 1e-300)
    out = {
        "pi_sorted": "|".join(f"{v:.4f}" for v in pi[order]),
        "mean_sorted": "|".join(f"{v:.4f}" for v in means[order]),
        "sd_sorted": "|".join(f"{v:.4f}" for v in sds[order]),
        "stay_sorted": "|".join(f"{v:.4f}" for v in stay[order]),
        "margins": ";".join(
            f"{b['i']}{b.get('j', '')}:{b['dist']}" for b in model.margin_blocks()),
        "copulas": "",
    }
    if model.variant.uses_copula:
        out["copulas"] = ";".join(
            f"{b['i']}{b['j']}:{b['name']}({float(b.get('tau', np.nan)):.3f})"
            for b in model.copula_blocks())
    vals = [v for b in model.margin_blocks() for v in b["params"].values()]
    vals += [float(b.get("tau", 0.0)) for b in (model.copula_blocks() if model.variant.uses_copula else [])]
    out["finite_params"] = bool(np.all(np.isfinite(np.asarray(vals, float))) and np.all(np.isfinite(pi)))
    return out


def min_margin_sd(model: PMCModel) -> float:
    """Smallest standard deviation over every margin block (state or pair)."""
    sds = []
    for b in model.margin_blocks():
        m = model.margin(int(b["i"])) if "j" not in b else model.margin(int(b["i"]), int(b["j"]))
        sds.append(float(m._frozen.std()))
    return float(np.nanmin(sds))


def n_free_params(model: PMCModel) -> int:
    """Free parameters: prior, margin parameters and copula parameters."""
    K = model.K
    k = K * (K - 1) if model.variant.has_markov_prior else K * (K + 1) // 2 - 1
    k += sum(len(b["params"]) for b in model.margin_blocks())
    if model.variant.uses_copula:
        for b in model.copula_blocks():
            k += 1 + sum(1 for key in ("df", "delta") if key in b)
    return int(k)


class _WarningCollector(logging.Handler):
    """Collects the WARNING records of the library during one task."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _collect_warnings():
    h = _WarningCollector()
    lg = logging.getLogger("pmcprg")
    lg.addHandler(h)
    lg.setLevel(logging.WARNING)
    lg.propagate = False
    return h


def _release(h):
    logging.getLogger("pmcprg").removeHandler(h)


def _summarise_warnings(msgs: list[str]) -> tuple[int, int, str]:
    """(number of warnings, of which 'regressed' ones, first distinct kinds)."""
    regress = sum("regressed" in m for m in msgs)
    kinds = []
    for m in msgs:
        head = m.split(":")[0][:60] if "regressed" in m else m[:120]
        if head not in kinds:
            kinds.append(head)
    return len(msgs), regress, " || ".join(kinds[:3])


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

_DATA_CACHE: dict = {}


def load_tsnh4(data_dir: str, quick: bool) -> dict:
    """tsNH4 (log scale): Y with the real NaN, Y_true at every position."""
    key = ("tsnh4", data_dir, quick)
    if key not in _DATA_CACHE:
        df = pd.read_csv(Path(data_dir) / "imputeTS/output/tsNH4_study_log.csv")
        if quick:
            df = df.iloc[:1500]
        Y = df["Y"].to_numpy(float)
        _DATA_CACHE[key] = {"Y": Y, "Y_true": df["Y_true"].to_numpy(float),
                            "datetime": df["datetime"].to_numpy()}
    return _DATA_CACHE[key]


def load_beijing(data_dir: str, station: str, quick: bool) -> dict:
    """One-year PM2.5 extract of a station with its real NaN.

    PM2.5 is reported in whole µg/m³ (11–12 % of consecutive hours are equal,
    402 hours at the 3 µg/m³ floor in Huairou): on ln(PM2.5) itself a state
    collapses onto the atom ln 3 with a vanishing variance and an unbounded
    likelihood (see the ``huairou_rawlog`` fits). The fitted series is
    therefore dequantised: ``Y = ln(PM2.5 + U(−0.5, 0.5))`` with a fixed seed
    per station. ``Y_lograw = ln(PM2.5)`` (the file's ``Y``) is the ground
    truth of every imputation and forecast score.
    """
    key = ("beijing", station, data_dir, quick)
    if key not in _DATA_CACHE:
        df = pd.read_csv(Path(data_dir) / f"beijing_air_quality/output/{station}_pm25_study.csv")
        if quick:
            df = df.iloc[:1500]
        raw = df["PM25_raw"].to_numpy(float)
        rng = np.random.default_rng([DEQUANT_SEED, sum(map(ord, station))])
        jit = rng.uniform(-0.5, 0.5, raw.size)
        _DATA_CACHE[key] = {"Y": np.log(raw + jit), "Y_lograw": df["Y"].to_numpy(float),
                            "datetime": df["datetime"].to_numpy()}
    return _DATA_CACHE[key]


def prep_pamap2(data_dir: str, subject: int, quick: bool) -> dict:
    """2 Hz log-SD series of the hand accelerometer norm, labels and gap masks.

    * Windows of 50 consecutive samples (0.5 s), aligned on the first row of
      ``subject10x_100Hz.csv``; the incomplete last window is dropped.
    * ``Y_complete`` — log of the standard deviation (ddof 0) of the hand
      ‖a‖ (column ``Y``, ±16 g accelerometer, gravity included) over the
      valid samples of the window: the intensity of the hand motion. A window
      with fewer than 2 valid samples has its NaN samples linearly
      interpolated first (1 window in the three subjects).
    * ``labels`` — majority activityID of the window (ties: smallest id).
    * ``real`` — window missing when ≥ 5 of its 50 samples are NaN (≥ 10 %);
      ``real_any`` — when ≥ 1 is (sensitivity rule).
    * ``Y_gapped`` — ``Y_complete`` with the ``real`` windows and MCAR blocks
      (10 % of the series, blocks of 20 windows = 10 s, seed = subject)
      removed; ``mcar`` flags the added blocks only.
    """
    key = ("pamap2", subject, data_dir, quick)
    if key in _DATA_CACHE:
        return _DATA_CACHE[key]
    df = pd.read_csv(Path(data_dir) / f"pamap2/output/subject{subject}_100Hz.csv",
                     usecols=["t", "X", "Y"])
    y = df["Y"].to_numpy(float)
    x = df["X"].to_numpy(int)
    w = PAMAP2_WINDOW
    n = len(y) // w
    yy = y[: n * w].reshape(n, w)
    xx = x[: n * w].reshape(n, w)
    miss_s = ~np.isfinite(yy)
    cnt = miss_s.sum(axis=1)
    valid = w - cnt
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = np.nanstd(np.where(miss_s, np.nan, yy), axis=1)
    bad = valid < 2
    if bad.any():
        t = np.arange(len(y))
        fin = np.isfinite(y)
        y_lin = np.interp(t, t[fin], y[fin])[: n * w].reshape(n, w)
        sd[bad] = y_lin[bad].std(axis=1)
    Yc = np.log(sd)
    ids = np.unique(xx)
    counts = np.stack([(xx == a).sum(axis=1) for a in ids], axis=1)
    labels = ids[np.argmax(counts, axis=1)]
    purity = counts.max(axis=1) / w
    if quick:  # smoke run: every 5th window, so that every activity group is present
        Yc, labels, purity, cnt, miss_s = Yc[::5], labels[::5], purity[::5], cnt[::5], miss_s[::5]
    g2 = np.array([GROUPS_K2.get(int(a), -1) for a in labels])
    g3 = np.array([GROUPS_K3.get(int(a), -1) for a in labels])
    real = cnt >= PAMAP2_MIN_MISSING
    real_any = cnt >= 1
    Y_real = Yc.copy()
    Y_real[real] = np.nan
    Y_gapped, mcar = patterns.mcar(Y_real, PAMAP2_MCAR_RATE, block_size=PAMAP2_MCAR_BLOCK,
                                   seed=subject)
    out = {
        "Y_complete": Yc, "Y_real": Y_real, "Y_gapped": Y_gapped,
        "labels": labels, "purity": purity, "g2": g2, "g3": g3,
        "real": real, "real_any": real_any, "mcar": mcar.astype(bool),
        "missing": real | mcar.astype(bool), "count_missing": cnt,
        "sample_missing": miss_s,
    }
    _DATA_CACHE[key] = out
    return out


def series_data(name: str, data_dir: str, quick: bool) -> np.ndarray:
    """Observation vector of a named series."""
    if name == "tsnh4":
        return load_tsnh4(data_dir, quick)["Y"]
    if name in ("huairou", "aotizhongxin"):
        return load_beijing(data_dir, name, quick)["Y"]
    if name.endswith("_rawlog"):
        return load_beijing(data_dir, name.split("_")[0], quick)["Y_lograw"]
    if name.startswith("pamap2_"):
        _, subj, which = name.split("_")
        return prep_pamap2(data_dir, int(subj), quick)[f"Y_{which}"]
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def initial_model(kind: str, K: int, Y: np.ndarray) -> PMCModel:
    """Declared starting model (its values are replaced by the k-means warm start)."""
    q = np.nanquantile(Y, (np.arange(K) + 0.5) / K)
    s = float(np.nanstd(Y)) / K
    raw: dict = {"model": {"name": f"{kind} K={K}", "K": K, "N_default": int(len(Y))}}
    if kind == "hmc_in":
        raw["model"]["variant"] = "HMC-IN"
        A = np.full((K, K), 0.1 / (K - 1))
        np.fill_diagonal(A, 0.9)
        raw["prior"] = {"A": A.tolist()}
    else:
        raw["model"]["variant"] = "PMC"
        p = np.full((K, K), 0.1 / (K - 1))
        np.fill_diagonal(p, 0.9)
        raw["prior"] = {"p": (p / p.sum()).tolist()}
    extra = {"candidates": list(GICE_CANDIDATES)} if kind == "pmc_state_gice" else {}
    if kind == "pmc_pair":
        raw["model"]["margin_structure"] = "pair"
        raw["margins"] = [{"i": i, "j": j, "dist": "norm",
                           "params": {"loc": float(q[i]), "scale": s}}
                          for i in range(K) for j in range(K)]
    else:
        raw["margins"] = [{"i": i, "dist": "norm", "params": {"loc": float(q[i]), "scale": s}, **extra}
                          for i in range(K)]
    if kind != "hmc_in":
        raw["copulas"] = [{"i": i, "j": j, "name": "Gauss", "tau": 0.3}
                          for i in range(K) for j in range(K)]
    return PMCModel.from_dict(raw)


def margin_rule(kind: str) -> str:
    return GICE_RULE if kind == "pmc_state_gice" else "mle"


def starting_model(kind: str, K: int, Y: np.ndarray, start: int, n_starts: int) -> tuple[PMCModel, str]:
    """Start 0: k-means warm start; start s ≥ 1: library multistart draw s from it.

    Copula models redraw the family of every pair among the candidates
    (``multistart_families = "random"``) on top of the parameter jitter.
    """
    m0 = initial_model(kind, K, Y)
    km = warmstart_from_kmeans(m0, Y, random_state=0, fit_margins=True,
                               candidates=list(COPULA_CANDIDATES),
                               selection_criterion="mle",
                               margin_selection_rule=margin_rule(kind))
    if start == 0:
        return km, "kmeans"
    cfg = {"n_starts": max(n_starts, start + 1), "multistart_seed": START_SEED,
           "multistart_jitter": START_JITTER,
           "multistart_families": "random" if km.variant.uses_copula else "none",
           "candidates": list(COPULA_CANDIDATES)}
    init, tag = build_multistart_inits(km, cfg, label="A4")[start]
    return init, tag


def estimate(init: PMCModel, Y: np.ndarray, kind: str, strategy: str, start: int):
    """One unsupervised fit (single start). Returns (model, trace)."""
    cfg = {"fit_margins": True, "candidates": list(COPULA_CANDIDATES),
           "selection_criterion": "mle", "margin_selection_rule": margin_rule(kind),
           "init": "model", "n_starts": 1}
    if strategy == "sem":
        cfg.update(max_iter=SEM_MAX_ITER, sem_seed=1000 + start)
        return sem(init, Y, sem_cfg=cfg)
    cfg.update(max_iter=ICE_MAX_ITER, tol=ICE_TOL)
    if strategy in ("available", "impute"):
        cfg.update(missing_strategy=strategy, missing_seed=1000 + start)
    return ice(init, Y, ice_cfg=cfg)


def supervised_model(kind: str, K: int, Y: np.ndarray, groups: np.ndarray) -> PMCModel:
    """Oracle: one supervised M-step on the labels (transient windows skipped).

    Uses the library M-step on one-hot posteriors with the observation mask
    ``groups >= 0``; run twice so that the copulas see the fitted margins.
    """
    N = len(Y)
    obs = (groups >= 0) & np.isfinite(Y)
    lab = np.where(obs, groups, 0)
    gamma = np.zeros((N, K))
    gamma[np.nonzero(obs)[0], lab[obs]] = 1.0
    both = np.nonzero(obs[:-1] & obs[1:])[0]
    xi = np.zeros((N - 1, K, K))
    xi[both, lab[both], lab[both + 1]] = 1.0
    raw = initial_model(kind, K, Y).raw
    for _ in range(2):
        current = PMCModel.from_dict(raw)
        m_step(raw, current, Y, xi, gamma, fit_margins=True,
               candidates=list(COPULA_CANDIDATES), selection_criterion="mle",
               margin_selection_rule=margin_rule(kind), obs=obs)
    # Transitions never seen between labelled windows (e.g. rest → vigorous,
    # always separated by transient windows) would be exact zeros: smoothed.
    key = "A" if "A" in raw["prior"] else "p"
    P = np.asarray(raw["prior"][key], float) + ORACLE_PRIOR_FLOOR
    P = P / P.sum(axis=1, keepdims=True) if key == "A" else P / P.sum()
    raw["prior"][key] = P.tolist()
    return PMCModel.from_dict(raw)


# ---------------------------------------------------------------------------
# Baseline imputations
# ---------------------------------------------------------------------------

def fill_linear(Y: np.ndarray) -> np.ndarray:
    """Linear interpolation, nearest observed value beyond the ends."""
    Y = np.asarray(Y, float)
    fin = np.isfinite(Y)
    t = np.arange(len(Y))
    out = Y.copy()
    out[~fin] = np.interp(t[~fin], t[fin], Y[fin])
    return out


def fill_locf(Y: np.ndarray) -> np.ndarray:
    """Last observation carried forward (first observed value carried backward)."""
    s = pd.Series(np.asarray(Y, float))
    return s.ffill().bfill().to_numpy()


def _ar1_nll(theta, y_obs, lag):
    mu, phi, log_s2 = theta[0], np.tanh(theta[1]), theta[2]
    s2 = np.exp(log_s2)
    v = s2 / (1.0 - phi ** 2)
    x = y_obs - mu
    rho = phi ** lag
    var = v * (1.0 - rho ** 2)
    var = np.maximum(var, 1e-12)
    resid = x[1:] - rho * x[:-1]
    ll = -0.5 * (np.log(2 * np.pi * v) + x[0] ** 2 / v)
    ll += -0.5 * np.sum(np.log(2 * np.pi * var) + resid ** 2 / var)
    return -ll


def ar1_fit(Y: np.ndarray) -> dict:
    """Exact Gaussian AR(1) maximum likelihood on the observed values (gaps allowed).

    y_t = μ + x_t, x_t = φ x_{t−1} + ε_t, ε_t ~ N(0, σ²), stationary start;
    two observed values k steps apart: x_{t+k} | x_t ~ N(φ^k x_t, v(1 − φ^{2k}))
    with v = σ² / (1 − φ²) — the likelihood of the Kalman filter of this
    state-space model, in closed form.
    """
    Y = np.asarray(Y, float)
    idx = np.nonzero(np.isfinite(Y))[0]
    y_obs = Y[idx]
    lag = np.diff(idx)
    mu0 = float(np.mean(y_obs))
    both = lag == 1
    x = y_obs - mu0
    r1 = float(np.sum(x[1:][both] * x[:-1][both]) / max(np.sum(x[:-1][both] ** 2), 1e-12))
    r1 = float(np.clip(r1, -0.95, 0.99))
    s2 = float(np.var(y_obs)) * (1 - r1 ** 2)
    theta0 = np.array([mu0, np.arctanh(r1), np.log(max(s2, 1e-8))])
    res = optimize.minimize(_ar1_nll, theta0, args=(y_obs, lag), method="L-BFGS-B")
    mu, phi, s2 = float(res.x[0]), float(np.tanh(res.x[1])), float(np.exp(res.x[2]))
    return {"mu": mu, "phi": phi, "sigma2": s2, "loglik": float(-res.fun)}


def ar1_smooth(Y: np.ndarray, par: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Posterior mean and sd of every missing value under the fitted AR(1).

    The AR(1) is Markov, so the Kalman (RTS) smoother at a missing t depends on
    the nearest observed values a < t < b only: a Gaussian bridge (one-sided
    at the ends). Returns (index, mean, sd).
    """
    Y = np.asarray(Y, float)
    N = len(Y)
    mu, phi, s2 = par["mu"], par["phi"], par["sigma2"]
    v = s2 / (1 - phi ** 2)
    fin = np.isfinite(Y)
    idx_obs = np.nonzero(fin)[0]
    miss = np.nonzero(~fin)[0]
    pos = np.searchsorted(idx_obs, miss)             # first observed index > t
    has_left = pos > 0
    has_right = pos < idx_obs.size
    a = np.where(has_left, idx_obs[np.maximum(pos - 1, 0)], 0)
    b = np.where(has_right, idx_obs[np.minimum(pos, idx_obs.size - 1)], N - 1)
    xa = np.where(has_left, Y[a] - mu, 0.0)
    xb = np.where(has_right, Y[b] - mu, 0.0)
    s = (miss - a).astype(float)
    d = (b - miss).astype(float)
    ps, pd_ = phi ** s, phi ** d
    mean = np.zeros(miss.size)
    var = np.full(miss.size, v)
    bi = has_left & has_right
    rho = phi ** (s + d)
    den = np.maximum(1 - rho ** 2, 1e-15)
    mean[bi] = ((ps - rho * pd_) * xa + (pd_ - rho * ps) * xb)[bi] / den[bi]
    var[bi] = v * (1 - (ps ** 2 - 2 * rho * ps * pd_ + pd_ ** 2)[bi] / den[bi])
    lo = has_left & ~has_right
    mean[lo] = (ps * xa)[lo]
    var[lo] = v * (1 - ps[lo] ** 2)
    ro = has_right & ~has_left
    mean[ro] = (pd_ * xb)[ro]
    var[ro] = v * (1 - pd_[ro] ** 2)
    return miss, mu + mean, np.sqrt(np.maximum(var, 1e-12))


def ar1_forecast(y_last: float, par: dict, h: int) -> tuple[np.ndarray, np.ndarray]:
    mu, phi, s2 = par["mu"], par["phi"], par["sigma2"]
    v = s2 / (1 - phi ** 2)
    k = np.arange(1, h + 1)
    return mu + phi ** k * (y_last - mu), np.sqrt(v * (1 - phi ** (2 * k)))


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

Z90 = 1.6448536269514722


def score_probabilistic(truth, mean, lo, hi, samples=None, sd=None) -> dict:
    """RMSE, MAE, CRPS, 90 % coverage and width on already-selected positions."""
    truth = np.asarray(truth, float)
    sel = np.ones(truth.size, bool)
    out = {"n": int(truth.size),
           "rmse": metrics.rmse(truth, mean, sel), "mae": metrics.mae(truth, mean, sel)}
    if samples is not None:
        out["crps"] = metrics.crps_from_samples(truth, samples)
        out["crps_kind"] = "samples"
    elif sd is not None:
        out["crps"] = metrics.crps_gaussian(truth, mean, sd)
        out["crps_kind"] = "gaussian"
    else:
        out["crps"] = out["mae"]
        out["crps_kind"] = "dirac(=mae)"
    if lo is not None:
        out["cov90"] = metrics.interval_coverage(truth, lo, hi)
        out["width90"] = float(np.mean(np.asarray(hi) - np.asarray(lo)))
    else:
        out["cov90"] = np.nan
        out["width90"] = np.nan
    return out


def split_errors(x_true: np.ndarray, x_hat: np.ndarray, K: int, keep: np.ndarray,
                 subsets: dict[str, np.ndarray]) -> dict:
    """Error rates on ``keep`` (non-transient) positions, one permutation for all.

    The permutation is the Hungarian one on all kept positions (as
    ``error_rate_split(align=True)``); ``subsets`` are boolean masks.
    """
    xt, xh = x_true[keep], align_labels(x_true[keep], x_hat[keep], K)
    wrong = xt != xh
    out = {"err_all": float(wrong.mean()), "n_all": int(keep.sum())}
    for name, m in subsets.items():
        mk = np.asarray(m, bool)[keep]
        out[f"err_{name}"] = float(wrong[mk].mean()) if mk.any() else np.nan
        out[f"n_{name}"] = int(mk.sum())
    miss = subsets.get("missing")
    if miss is not None and np.asarray(miss, bool)[keep].any():
        # the library's split, on the same (already aligned) labels
        er = metrics.error_rate_split(xt, xh, np.asarray(miss, bool)[keep], align=False)
        assert abs(er.overall - out["err_all"]) < 1e-12
    return out


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

def _save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))


def fit_tag(spec: dict) -> str:
    return f"{spec['series']}__K{spec['K']}__{spec['model']}__{spec['strategy']}__s{spec['start']}"


def task_fit(spec: dict) -> dict:
    """Unsupervised fit of one (series, K, model, strategy, start) + evaluations."""
    data_dir, quick, out = spec["data_dir"], spec["quick"], Path(spec["out"])
    name, K, kind, strategy, start = (spec["series"], spec["K"], spec["model"],
                                      spec["strategy"], spec["start"])
    Y = series_data(name, data_dir, quick)
    miss = ~np.isfinite(Y)
    row = {"series": name, "K": K, "model": kind, "strategy": strategy, "start": start,
           "N": int(len(Y)), "n_missing": int(miss.sum()), "status": "ok", "error": ""}
    h = _collect_warnings()
    t0 = time.perf_counter()
    try:
        init, tag = starting_model(kind, K, Y, start, spec["n_starts"])
        row["run_tag"] = tag
        t1 = time.perf_counter()
        fitted, trace = estimate(init, Y, kind, strategy, start)
        row["runtime_s"] = time.perf_counter() - t1
        row["init_s"] = t1 - t0
    except Exception as exc:  # recorded, the campaign goes on
        _release(h)
        row.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:300],
                   runtime_s=time.perf_counter() - t0)
        n_w, n_reg, kinds = _summarise_warnings(h.messages)
        row.update(n_warnings=n_w, n_regressions_warned=n_reg, warnings=kinds)
        return {"fits": [row]}
    lls = np.asarray(trace.log_liks, float)
    row["n_iter"] = int(lls.size)
    row["ll_trace_last"] = float(lls[-1]) if lls.size else np.nan
    row["n_ll_decreases"] = int(np.sum(np.diff(lls) < -1e-6)) if lls.size > 1 else 0
    row["max_ll_decrease"] = float(max(0.0, -np.min(np.diff(lls)))) if lls.size > 1 else 0.0
    x_hat, gamma, ll = classify(fitted, Y)
    row["ll"] = float(ll)
    row["n_params"] = n_free_params(fitted)
    n_obs = int((~miss).sum())
    row["bic"] = -2.0 * row["ll"] + row["n_params"] * np.log(n_obs)
    row.update(model_summary(fitted))
    _save_json(out / "models" / f"{fit_tag(spec)}.json", fitted.raw)
    np.savez_compressed(out / "labels" / f"{fit_tag(spec)}.npz",
                        x_hat=x_hat.astype(np.int8), conf=gamma.max(axis=1).astype(np.float32),
                        lls=lls)
    result = {"fits": [row]}
    evals = spec.get("evals", ())
    try:
        if "tsnh4_impute" in evals:
            result["imputation"] = _eval_tsnh4(fitted, spec, row)
        if "pamap2" in evals:
            result["classification"] = _eval_pamap2(fitted, spec, row, x_hat)
    except Exception as exc:
        row["eval_error"] = f"{type(exc).__name__}: {exc}"[:300]
    _release(h)
    n_w, n_reg, kinds = _summarise_warnings(h.messages)
    row.update(n_warnings=n_w, n_regressions_warned=n_reg, warnings=kinds)
    return result


def _eval_tsnh4(fitted: PMCModel, spec: dict, row: dict) -> list[dict]:
    d = load_tsnh4(spec["data_dir"], spec["quick"])
    Y, Yt = d["Y"], d["Y_true"]
    t0 = time.perf_counter()
    imp = impute(fitted, Y, quantiles=QUANTILES, n_samples=IMPUTE_DRAWS,
                 rng=np.random.default_rng(7))
    t_imp = time.perf_counter() - t0
    idx = imp.index
    gl = gap_lengths(~np.isfinite(Y))[idx]
    cls = gap_class(gl)
    rows = []
    base = {k: row[k] for k in ("series", "K", "model", "strategy", "start")}
    samples = imp.y_samples.T  # (M, S)
    for c in ("all", "1-2", "3-20", ">20"):
        sel = np.ones(idx.size, bool) if c == "all" else cls == c
        if not sel.any():
            continue
        sc = score_probabilistic(Yt[idx][sel], imp.mean[sel], imp.quantile_values[sel, 0],
                                 imp.quantile_values[sel, 2], samples=samples[sel])
        rows.append({**base, "method": "pmc_posterior", "gap_class": c, **sc,
                     "impute_s": t_imp})
    np.savez_compressed(Path(spec["out"]) / "imputations" / f"{fit_tag(spec)}.npz",
                        index=idx, mean=imp.mean, q=imp.quantile_values)
    return rows


def _eval_pamap2(fitted: PMCModel, spec: dict, row: dict, x_hat_fit: np.ndarray) -> list[dict]:
    """Classification errors of a PAMAP2 fit (complete or gapped series)."""
    _, subj, which = spec["series"].split("_")
    d = prep_pamap2(spec["data_dir"], int(subj), spec["quick"])
    K = spec["K"]
    groups = d["g2"] if K == 2 else d["g3"]
    keep = groups >= 0
    base = {k: row[k] for k in ("series", "K", "model", "strategy", "start")}
    base["subject"] = int(subj)
    base["fit_on"] = which
    subsets = {"missing": d["missing"], "observed": ~d["missing"], "real": d["real"],
               "mcar": d["mcar"]}
    rows = []
    if which == "complete":
        rows.append({**base, "applied_to": "complete", "method": "complete",
                     **split_errors(groups, x_hat_fit, K, keep, {})})
    else:
        xc, _, _ = classify(fitted, d["Y_complete"])
        rows.append({**base, "applied_to": "complete", "method": "complete",
                     **split_errors(groups, xc, K, keep, {})})
    rows.extend(gapped_methods(fitted, d, groups, K, keep, base, subsets))
    return rows


def gapped_methods(model, d, groups, K, keep, base, subsets) -> list[dict]:
    """Marginalisation vs fill-in-then-classify on the gapped series."""
    Yg = d["Y_gapped"]
    rows = []
    t0 = time.perf_counter()
    xm, gm, _ = classify(model, Yg)
    rows.append({**base, "applied_to": "gapped", "method": "marginalise",
                 **split_errors(groups, xm, K, keep, subsets),
                 "conf_missing": float(gm.max(axis=1)[d["missing"] & keep].mean()),
                 "time_s": time.perf_counter() - t0})
    imp = impute(model, Yg, quantiles=())
    fills = {"plugin": imp.Y_mean, "linear": fill_linear(Yg), "locf": fill_locf(Yg)}
    for meth, Yf in fills.items():
        xf, gf, _ = classify(model, Yf)
        rows.append({**base, "applied_to": "gapped", "method": meth,
                     **split_errors(groups, xf, K, keep, subsets),
                     "conf_missing": float(gf.max(axis=1)[d["missing"] & keep].mean())})
    return rows


def task_oracle(spec: dict) -> dict:
    """Supervised oracle (margins/prior/copulas from the labels) on PAMAP2."""
    subj, K, kind = spec["subject"], spec["K"], spec["model"]
    d = prep_pamap2(spec["data_dir"], subj, spec["quick"])
    groups = d["g2"] if K == 2 else d["g3"]
    keep = groups >= 0
    h = _collect_warnings()
    model = supervised_model(kind, K, d["Y_complete"], groups)
    base = {"series": f"pamap2_{subj}_oracle", "subject": subj, "K": K, "model": kind,
            "strategy": "supervised", "start": 0, "fit_on": "labels"}
    subsets = {"missing": d["missing"], "observed": ~d["missing"], "real": d["real"],
               "mcar": d["mcar"]}
    xc, _, llc = classify(model, d["Y_complete"])
    rows = [{**base, "applied_to": "complete", "method": "complete",
             **split_errors(groups, xc, K, keep, {}), "ll_complete": float(llc)}]
    rows.extend(gapped_methods(model, d, groups, K, keep, base, subsets))
    _release(h)
    summ = model_summary(model)
    for r in rows:
        r["copulas"] = summ["copulas"]
        r["mean_sorted"] = summ["mean_sorted"]
        r["n_warnings"] = len(h.messages)
    return {"oracle": rows}


def _hmm_posteriors(loglik_state: np.ndarray, A: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Log-space forward-backward of a K-state HMM from per-position log emissions."""
    N, K = loglik_state.shape
    with np.errstate(divide="ignore"):
        lA, lpi = np.log(A), np.log(pi)
    la = np.zeros((N, K))
    la[0] = lpi + loglik_state[0]
    for n in range(1, N):
        la[n] = logsumexp(la[n - 1][:, None] + lA, axis=0) + loglik_state[n]
    lb = np.zeros((N, K))
    for n in range(N - 2, -1, -1):
        lb[n] = logsumexp(lA + (loglik_state[n + 1] + lb[n + 1])[None, :], axis=1)
    lg = la + lb
    return np.exp(lg - logsumexp(lg, axis=1, keepdims=True))


def task_mnar(spec: dict) -> dict:
    """State-dependent missingness: ignorable vs missingness-aware HMC-IN oracle.

    Both models have the supervised Gaussian margins and transition matrix of
    the labels (HMC-IN). The aware one also has P(window missing | state) =
    r_k estimated from the labels, i.e. the emission at n is
    f_k(y_n) (1 − r_k) when y_n is observed and r_k when it is missing; the
    ignorable one uses 1 at a missing n (the library's rule). Series: the
    complete series with only the real gaps removed (rules ≥ 10 % and any).
    """
    subj = spec["subject"]
    d = prep_pamap2(spec["data_dir"], subj, spec["quick"])
    rows = []
    for K in ((2,) if spec["quick"] else (2, 3)):
        groups = d["g2"] if K == 2 else d["g3"]
        keep = groups >= 0
        oracle = supervised_model("hmc_in", K, d["Y_complete"], groups)
        A = np.asarray(oracle.transition_A, float)
        pi = np.asarray(oracle.stationary_pi, float)
        for rule in ("real", "real_any"):
            m = d[rule]
            Y = d["Y_complete"].copy()
            Y[m] = np.nan
            r = np.array([m[keep & (groups == k)].mean() for k in range(K)])
            r = np.clip(r, 1e-4, 1 - 1e-4)
            logf = np.stack([oracle.margin(k)._frozen.logpdf(np.where(m, 0.0, Y))
                             for k in range(K)], axis=1)
            le_ign = np.where(m[:, None], 0.0, logf)
            le_aw = np.where(m[:, None], np.log(r)[None, :], logf + np.log1p(-r)[None, :])
            _, lib_g, _ = classify(oracle, Y)
            for label, le in (("ignorable", le_ign), ("aware", le_aw)):
                g = _hmm_posteriors(le, A, pi)
                xh = g.argmax(axis=1)
                # the hand-written recursion, ignorable case, vs the library
                diff_lib = float(np.max(np.abs(g - lib_g))) if label == "ignorable" else np.nan
                err_m = float(np.mean(xh[m & keep] != groups[m & keep])) if (m & keep).any() else np.nan
                err_o = float(np.mean(xh[~m & keep] != groups[~m & keep]))
                truth_share = [float(np.mean(groups[m & keep] == k)) if (m & keep).any() else np.nan
                               for k in range(K)]
                post_share = [float(g[m & keep, k].mean()) if (m & keep).any() else np.nan
                              for k in range(K)]
                rows.append({"subject": subj, "K": K, "rule": rule, "model": label,
                             "n_missing": int((m & keep).sum()), "n_kept": int(keep.sum()),
                             "err_missing": err_m, "err_observed": err_o,
                             "miss_rate_by_state": "|".join(f"{v:.4f}" for v in r),
                             "truth_share_missing": "|".join(f"{v:.3f}" for v in truth_share),
                             "posterior_share_missing": "|".join(f"{v:.3f}" for v in post_share),
                             "stationary_pi": "|".join(f"{v:.3f}" for v in pi),
                             "max_abs_diff_vs_library": diff_lib})
    return {"mnar": rows}


def task_pamap2_missingness(spec: dict) -> dict:
    """Per-activity and per-group missing rates (100 Hz samples and 2 Hz windows)."""
    subj = spec["subject"]
    d = prep_pamap2(spec["data_dir"], subj, spec["quick"])
    lab = d["labels"]
    rows = []
    levels = [("activity", lab, {a: PAMAP2_ACTIVITIES.get(int(a), "transient") for a in np.unique(lab)}),
              ("K3", d["g3"], {-1: "transient", 0: "rest", 1: "locomotion", 2: "vigorous"}),
              ("K2", d["g2"], {-1: "transient", 0: "rest", 1: "active"})]
    for level, codes, names in levels:
        for c in np.unique(codes):
            s = codes == c
            rows.append({"subject": subj, "level": level, "code": int(c), "name": names[int(c)],
                         "n_windows": int(s.sum()),
                         "sample_nan_pct": 100 * float(d["sample_missing"][s].mean()),
                         "win_missing_pct_ge10": 100 * float(d["real"][s].mean()),
                         "win_missing_pct_any": 100 * float(d["real_any"][s].mean()),
                         "mcar_pct": 100 * float(d["mcar"][s].mean()),
                         "mean_Y": float(np.mean(d["Y_complete"][s])),
                         "sd_Y": float(np.std(d["Y_complete"][s]))})
    Yg = d["Y_gapped"]
    info = {"subject": subj, "level": "series", "code": -9, "name": "all",
            "n_windows": int(len(Yg)),
            "sample_nan_pct": 100 * float(d["sample_missing"].mean()),
            "win_missing_pct_ge10": 100 * float(d["real"].mean()),
            "win_missing_pct_any": 100 * float(d["real_any"].mean()),
            "mcar_pct": 100 * float(d["mcar"].mean()),
            "mean_Y": float(np.mean(d["Y_complete"])), "sd_Y": float(np.std(d["Y_complete"]))}
    rows.append(info)
    strip = None
    if spec.get("save_strip"):
        strip = {"t_s": np.arange(len(Yg)) * PAMAP2_WINDOW / 100.0, "Y": d["Y_complete"],
                 "g3": d["g3"], "real": d["real"].astype(int), "mcar": d["mcar"].astype(int)}
        np.savez_compressed(Path(spec["out"]) / f"pamap2_{subj}_strip.npz", **strip)
    return {"pamap2_missingness": rows}


def task_pamap2_feature_check(spec: dict) -> dict:
    """Why the log-SD feature and this K = 3 grouping: supervised HMC-IN errors.

    Features: log-SD (used) and mean of the hand ‖a‖ over each window.
    Groupings: K2, K3 (used) and K3 with ascending stairs vigorous and Nordic
    walking locomotion (``K3_alt``). Complete series, no gaps.
    """
    subj = spec["subject"]
    d = prep_pamap2(spec["data_dir"], subj, spec["quick"])
    df = pd.read_csv(Path(spec["data_dir"]) / f"pamap2/output/subject{subj}_100Hz.csv", usecols=["Y"])
    y = df["Y"].to_numpy(float)
    w = PAMAP2_WINDOW
    n = len(y) // w
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(y[: n * w].reshape(n, w), axis=1)
    if spec["quick"]:
        mean = mean[::5]
    fin = np.isfinite(mean)
    mean[~fin] = np.interp(np.nonzero(~fin)[0], np.nonzero(fin)[0], mean[fin])
    alt = dict(GROUPS_K3)
    alt.update({12: 2, 7: 1})
    groupings = {"K2": (2, d["g2"]), "K3": (3, d["g3"]),
                 "K3_alt": (3, np.array([alt.get(int(a), -1) for a in d["labels"]]))}
    rows = []
    for feat, Y in (("log_sd", d["Y_complete"]), ("mean", mean)):
        for gname, (K, groups) in groupings.items():
            model = supervised_model("hmc_in", K, Y, groups)
            xh, _, _ = classify(model, Y)
            keep = groups >= 0
            rows.append({"subject": subj, "feature": feat, "grouping": gname, "K": K,
                         "err": float(np.mean(xh[keep] != groups[keep])), "n": int(keep.sum())})
    return {"feature_check": rows}


def task_tsnh4_baselines(spec: dict) -> dict:
    """Linear, LOCF and AR(1) smoother at the real gaps of tsNH4."""
    d = load_tsnh4(spec["data_dir"], spec["quick"])
    Y, Yt = d["Y"], d["Y_true"]
    miss = ~np.isfinite(Y)
    idx = np.nonzero(miss)[0]
    cls = gap_class(gap_lengths(miss)[idx])
    t0 = time.perf_counter()
    par = ar1_fit(Y)
    ai, am, asd = ar1_smooth(Y, par)
    t_ar = time.perf_counter() - t0
    assert np.array_equal(ai, idx)
    preds = {"linear": (fill_linear(Y)[idx], None), "locf": (fill_locf(Y)[idx], None),
             "ar1": (am, asd)}
    rows = []
    for meth, (mean, sd) in preds.items():
        for c in ("all", "1-2", "3-20", ">20"):
            sel = np.ones(idx.size, bool) if c == "all" else cls == c
            if not sel.any():
                continue
            if sd is None:
                sc = score_probabilistic(Yt[idx][sel], mean[sel], None, None)
            else:
                sc = score_probabilistic(Yt[idx][sel], mean[sel], mean[sel] - Z90 * sd[sel],
                                         mean[sel] + Z90 * sd[sel], sd=sd[sel])
            rows.append({"series": "tsnh4", "K": 0, "model": meth, "strategy": "baseline",
                         "start": 0, "method": meth, "gap_class": c, **sc,
                         "impute_s": t_ar if meth == "ar1" else 0.0,
                         "ar1_params": json.dumps(par) if meth == "ar1" else ""})
    np.savez_compressed(Path(spec["out"]) / "imputations" / "tsnh4_baselines.npz",
                        index=idx, linear=preds["linear"][0], locf=preds["locf"][0],
                        ar1_mean=am, ar1_sd=asd)
    return {"imputation": rows}


def task_cv(spec: dict) -> dict:
    """Masked cross-validation on a Beijing station: hide observed values, refit, impute."""
    st, rate, block, seed = spec["station"], spec["rate"], spec["block"], spec["seed"]
    bj = load_beijing(spec["data_dir"], st, spec["quick"])
    Y = bj["Y"]
    ss = [seed, int(1000 * rate), block, sum(map(ord, st))]
    Ym, mask = patterns.mcar(Y, rate, block_size=block, seed=np.random.default_rng(ss))
    mask = mask.astype(bool)
    truth = bj["Y_lograw"][mask]
    base = {"station": st, "rate": rate, "block": block, "seed": seed,
            "n_hidden": int(mask.sum()), "n_real_missing": int((~np.isfinite(Y)).sum())}
    rows = []
    if spec["what"] == "baselines":
        for meth, Yf in (("linear", fill_linear(Ym)), ("locf", fill_locf(Ym))):
            rows.append({**base, "method": meth, "K": 0,
                         **score_probabilistic(truth, Yf[mask], None, None)})
        t0 = time.perf_counter()
        par = ar1_fit(Ym)
        ai, am, asd = ar1_smooth(Ym, par)
        pos = np.searchsorted(ai, np.nonzero(mask)[0])
        sc = score_probabilistic(truth, am[pos], am[pos] - Z90 * asd[pos], am[pos] + Z90 * asd[pos],
                                 sd=asd[pos])
        rows.append({**base, "method": "ar1", "K": 0, **sc, "fit_s": time.perf_counter() - t0})
        return {"cv": rows}
    kind, K = spec["model"], spec["K"]
    h = _collect_warnings()
    t0 = time.perf_counter()
    try:
        init, _ = starting_model(kind, K, Ym, 0, 1)
        fitted, trace = estimate(init, Ym, kind, "available", 0)
    except Exception as exc:
        _release(h)
        rows.append({**base, "method": kind, "K": K, "status": "failed",
                     "error": f"{type(exc).__name__}: {exc}"[:300]})
        return {"cv": rows}
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    imp = impute(fitted, Ym, quantiles=QUANTILES, n_samples=IMPUTE_DRAWS,
                 rng=np.random.default_rng(ss + [7]))
    imp_s = time.perf_counter() - t1
    pos = np.searchsorted(imp.index, np.nonzero(mask)[0])
    sc = score_probabilistic(truth, imp.mean[pos], imp.quantile_values[pos, 0],
                             imp.quantile_values[pos, 2], samples=imp.y_samples.T[pos])
    x_cv, _, ll = classify(fitted, Ym)
    ref = np.load(spec["reference_labels"])["x_hat"].astype(int)
    fin = np.isfinite(Ym)
    _release(h)
    rows.append({**base, "method": kind, "K": K, "status": "ok", **sc, "fit_s": fit_s,
                 "impute_s": imp_s, "n_iter": len(trace.log_liks), "ll": float(ll),
                 "agree_ref_observed": agreement(ref, x_cv, K, fin),
                 "agree_ref_hidden": agreement(ref, x_cv, K, mask),
                 "n_warnings": len(h.messages), **model_summary(fitted)})
    return {"cv": rows}


def task_forecast(spec: dict) -> dict:
    """Rolling-origin h-step forecasts on a Beijing station vs persistence and AR(1).

    The models are fitted (ICE, available, k-means start) on the first
    ``train_frac`` of the series; origins every ``step`` hours in the rest.
    The forecast at an origin conditions on every observation before it.
    """
    st, kinds = spec["station"], spec["models"]
    bj = load_beijing(spec["data_dir"], st, spec["quick"])
    Y, Y_truth = bj["Y"], bj["Y_lograw"]
    N = len(Y)
    n_train = int(spec["train_frac"] * N)
    H, step = spec["h"], spec["step"]
    origins = np.arange(n_train, N - H, step)
    h = _collect_warnings()
    fitted = {}
    for kind, K in kinds:
        init, _ = starting_model(kind, K, Y[:n_train], 0, 1)
        fitted[(kind, K)], _ = estimate(init, Y[:n_train], kind, "available", 0)
    par = ar1_fit(Y[:n_train])
    rows = []
    for o in origins:
        fut = Y_truth[o: o + H]
        prefix = Y[:o]
        fin = np.nonzero(np.isfinite(prefix))[0]
        last = prefix[fin[-1]]
        preds = {"persistence": (np.full(H, last), None, None, None)}
        am, asd = ar1_forecast(last, par, H + (o - 1 - fin[-1]))
        am, asd = am[-H:], asd[-H:]
        preds["ar1"] = (am, asd, am - Z90 * asd, am + Z90 * asd)
        for (kind, K), mdl in fitted.items():
            fc = forecast(mdl, prefix, H, quantiles=QUANTILES)
            preds[f"{kind}_K{K}"] = (fc.mean, fc.sd, fc.quantile_values[:, 0], fc.quantile_values[:, 2])
        for meth, (mean, sd, lo, hi) in preds.items():
            for k in range(H):
                if not np.isfinite(fut[k]):
                    continue
                rows.append({"station": st, "origin": int(o), "h": k + 1, "method": meth,
                             "y": float(fut[k]), "mean": float(mean[k]),
                             "sd": float(sd[k]) if sd is not None else np.nan,
                             "lo90": float(lo[k]) if lo is not None else np.nan,
                             "hi90": float(hi[k]) if hi is not None else np.nan})
    _release(h)
    df = pd.DataFrame(rows)
    agg = []
    for (meth, hh), g in df.groupby(["method", "h"]):
        r = {"station": st, "method": meth, "h": int(hh), "n": len(g),
             "rmse": float(np.sqrt(np.mean((g.y - g["mean"]) ** 2))),
             "mae": float(np.mean(np.abs(g.y - g["mean"])))}
        if g.sd.notna().all():
            r["crps_gauss"] = metrics.crps_gaussian(g.y.to_numpy(), g["mean"].to_numpy(), g.sd.to_numpy())
            r["cov90"] = metrics.interval_coverage(g.y.to_numpy(), g.lo90.to_numpy(), g.hi90.to_numpy())
        agg.append(r)
    return {"forecast": agg}


TASKS = {"fit": task_fit, "oracle": task_oracle, "mnar": task_mnar,
         "pamap2_missingness": task_pamap2_missingness, "tsnh4_baselines": task_tsnh4_baselines,
         "pamap2_feature_check": task_pamap2_feature_check,
         "cv": task_cv, "forecast": task_forecast}


def run_task(spec: dict) -> tuple[dict, dict, float]:
    """Worker entry point: (spec, result, wall seconds)."""
    logging.getLogger("pmcprg").setLevel(logging.WARNING)
    t0 = time.perf_counter()
    try:
        res = TASKS[spec["task"]](spec)
    except Exception as exc:  # a crashed task is recorded, the campaign goes on
        import traceback
        res = {"task_errors": [{"task": spec["task"],
                                "spec": json.dumps({k: v for k, v in spec.items()
                                                    if k not in ("data_dir", "out")}, default=str),
                                "error": f"{type(exc).__name__}: {exc}"[:500],
                                "traceback": traceback.format_exc()[-1500:]}]}
    return spec, res, time.perf_counter() - t0
