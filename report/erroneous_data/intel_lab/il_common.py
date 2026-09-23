"""
il_common.py — data, models, baselines and scores of the Intel Lab study of
erroneous-data detection (:mod:`pmcprg.pmc.outliers` on real sensor data).

The data are read from ``--data`` (default: the ``intel_lab`` folder next to
the repository) and never copied into the repository. Conventions (model
builders, k-means starts, multistart draws, summaries, label alignment) are
those of ``report/real_series/rs_common.py``, imported from there.

Every choice below is a named constant, quoted by the README.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT, ROOT / "report" / "real_series"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pmcprg  # noqa: E402
import rs_common as rc  # noqa: E402
from pmcprg.pmc import PMCModel, classify, ice  # noqa: E402

DEFAULT_DATA = "/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/data/timeseries/intel_lab"
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
CACHE = RESULTS / "cache"          # per-row flags, fitted fits of step 3 (not versioned)
MODELS_DIR = RESULTS / "models"    # the clean-window fits (TOML, versioned: small)

# ---------------------------------------------------------------------------
# Data choices
# ---------------------------------------------------------------------------

#: The study selection of the dataset README (best-covered motes).
MOTES = (48, 22, 47)
#: Temperature resolution of the readings (smallest step between distinct
#: values, measured: 0.0098 °C). A third of the 30-s increments are exactly 0,
#: so the one-step predictive law is discrete at the scale the copula models
#: resolve: the fitted series is dequantised (randomised PIT, as the Beijing
#: PM2.5 series of report/real_series).
QUANTUM = 0.0098
#: Seed of the dequantisation jitter U(−QUANTUM/2, QUANTUM/2): [DEQUANT_SEED, mote].
DEQUANT_SEED = 11
#: Nominal time of epoch e: T_REF + ORIGIN_S + P_S · e (epoch_time_mapping.json).
T_REF = pd.Timestamp("2004-02-28 00:00:00")
P_S = 30.000501093777807
ORIGIN_S = 3472.2380764110103
EPH = 3600.0 / P_S                 # epochs per hour (≈ 120)
EPD = 24.0 * EPH                   # epochs per day (≈ 2880)
#: Clean window of the README: epochs 1–28 800 (10 days), no suspect reading
#: for the three motes. Fit on days 1–7, hold out days 8–10.
FIT_END = 20160
CLEAN_END = 28800
#: Suspect rule of the dataset (temperature ∉ [−10, 60] °C or humidity ∉
#: [0, 100] %); the temperature half is the fixed-threshold reference.
T_LO, T_HI = -10.0, 60.0

# ---------------------------------------------------------------------------
# Model choices
# ---------------------------------------------------------------------------

#: HMC-IN, and PMC with state margins (= stationary reversible HMC-DN: the
#: copula variant of report/real_series, "pmc_state").
KINDS = ("hmc_in", "pmc_state")
KIND_LABEL = {"hmc_in": "HMC-IN", "pmc_state": "PMC"}
KS = (2, 3)
#: Starts per (mote, kind, K): k-means + 2 library multistart draws (rs_common).
N_STARTS = 3
ICE_MAX_ITER = 50
ICE_TOL = 1e-4
#: Detection settings.
ALPHAS = (1e-2, 1e-3, 1e-4)
BH_ALPHAS = (1e-2, 1e-3)
MAIN_ALPHA = 1e-3
#: Quadrature nodes of the missing rows (grid variants: the PMC): the
#: library default, used by every fit, PIT and flag of the study.
GAP_NODES = 64
#: The sensitivity check of the PMC computations (``--gap-nodes`` of
#: fit_clean.py and detect.py): the node count at which the library's
#: quadrature diagnostic falls below its WARNING threshold on this data.
GAP_NODES_CHECK = 256

# ---------------------------------------------------------------------------
# Baselines and scoring choices
# ---------------------------------------------------------------------------

#: Hampel identifier (centred window of 2·HAMPEL_HALF + 1 observed rows, as
#: report/erroneous_data), thresholds in robust sd; MAD floored at one quantum
#: (the dequantised MAD of a flat window is ≈ 0.004 °C).
HAMPEL_HALF = 5
HAMPEL_TS = (3.0, 4.0, 5.0)
#: Rolling robust z-score: trailing window of 24 h of observed rows (the
#: current row excluded), at least ROBZ_MIN rows, thresholds in robust sd.
ROBZ_WINDOW_H = 24.0
ROBZ_MIN = 30
ROBZ_TS = (3.0, 4.0, 5.0)
#: Fixed upper thresholds below the 60 °C of the suspect rule.
FIXED_TS = (35.0, 40.0, 50.0)
#: Clean-envelope threshold: outside [min, max] of the fit window ± this margin.
ENVELOPE_MARGIN = 2.0
#: Alarm episodes: flags less than EPISODE_GAP_H apart belong to one episode.
EPISODE_GAP_H = 1.0
#: Zone before the first threshold hit whose flags are "ambiguous" (possible
#: early drift) rather than false alarms: the last AMBIG_H hours.
AMBIG_H = 24.0
#: Rolling median used to date the out-of-range climb (trailing, hours).
CLIMB_MEDIAN_H = 1.0


#: Figure colours (reference categorical palette, fixed order) and ink.
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C_MAGENTA, C_GREEN, C_VIOLET, C_RED = "#e87ba4", "#008300", "#4a3aa7", "#e34948"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def plot_setup():
    """matplotlib with the package look, thin recessive axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pmcprg.plot_style import apply_style
    apply_style(font_size=9)
    plt.rcParams.update({"axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "lines.linewidth": 1.0,
                         "legend.frameon": False})
    return plt


def check_import() -> str:
    """Path of the imported package; it must be this checkout's."""
    path = str(Path(pmcprg.__file__).resolve())
    if not path.startswith(str(ROOT)):
        raise RuntimeError(f"pmcprg imported from {path}, not from {ROOT}")
    return path


def git_commit() -> str:
    """The commit of this checkout (for the *_info.json files); "+dirty" if pmcprg/ is modified."""
    import subprocess
    try:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "pmcprg"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return head + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class _WarningLog(logging.Handler):
    """Keeps the messages of the WARNING records of ``pmcprg`` in a worker."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def capture_warnings() -> _WarningLog:
    """Collect (instead of printing) the WARNINGs of ``pmcprg`` from now on.

    Called at the start of every worker task: the quadrature WARNING of the
    missing-data passes ("Missing-data quadrature not converged …",
    ``GapPosterior.quad_error`` above its threshold) is logged by every pass
    that integrates a gap (each ICE E-step, ``classify``), and
    :func:`warning_summary` counts it per task for the result tables.
    """
    lg = logging.getLogger("pmcprg")
    for h in list(lg.handlers):
        if isinstance(h, _WarningLog):
            lg.removeHandler(h)
    h = _WarningLog()
    lg.addHandler(h)
    lg.setLevel(logging.WARNING)
    lg.propagate = False
    return h


_QUAD_RE = re.compile(r"Missing-data quadrature not converged: relative error "
                      r"([0-9.eE+-]+) > ([0-9.eE+-]+)")


def warning_summary(log: _WarningLog) -> dict:
    """Counts of the captured WARNINGs: the quadrature ones and the others."""
    quad, lim, other = [], [], []
    for msg in log.messages:
        m = _QUAD_RE.search(msg)
        if m:
            quad.append(float(m.group(1)))
            lim.append(float(m.group(2)))
        else:
            other.append(msg.split(";")[0][:160])
    return {"quad_warnings": len(quad),
            "quad_error_max_warned": max(quad) if quad else np.nan,
            "quad_limit_warned": max(lim) if lim else np.nan,
            "other_warnings": len(other),
            "other_warning_kinds": " || ".join(sorted(set(other)))}


def quad_check(model: PMCModel, Y: np.ndarray, gap_nodes: int = GAP_NODES) -> dict:
    """The library's quadrature diagnostic of one missing-data pass over Y.

    ``gap_posterior`` builds the grids that ``predictive_pit`` builds on the
    same missing rows (its log-likelihood is the PIT filter's). Returns
    ``quad_error``, its WARNING threshold and the log-likelihood; NaN for a
    model without a grid (HMC-IN) or a Y without gaps.
    """
    from pmcprg.pmc import gaps
    miss = ~np.isfinite(np.asarray(Y, float))
    post = gaps.gap_posterior(model, Y, gap_nodes=gap_nodes, xi=False)
    qe = post.quad_error
    return {"quad_error": float(qe) if qe is not None else np.nan,
            "quad_limit": float(gaps.quad_warn_limit(miss)) if qe is not None else np.nan,
            "log_lik": float(post.log_lik)}


def epoch_time(epoch) -> pd.DatetimeIndex:
    """Nominal time of the epochs (dataset mapping)."""
    e = np.asarray(epoch, float)
    return T_REF + pd.to_timedelta(ORIGIN_S + P_S * e, unit="s")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

_CACHE: dict = {}


def load_mote(mote: int, data_dir: str = DEFAULT_DATA) -> dict:
    """One mote on the full epoch grid (1 … 102 706).

    ``y`` is the dequantised raw temperature (suspect readings kept), NaN
    where no reading was received; ``y_clean`` is the same with the suspect
    readings set to NaN (the file's ``Y``, dequantised).
    """
    key = (mote, data_dir)
    if key in _CACHE:
        return _CACHE[key]
    df = pd.read_csv(Path(data_dir) / "output" / f"mote{mote}_epochs.csv",
                     usecols=["epoch", "temperature", "humidity", "voltage",
                              "suspect", "Y_raw", "Y"])
    epoch = df["epoch"].to_numpy(int)
    if not np.array_equal(epoch, np.arange(1, epoch.size + 1)):
        raise ValueError("epoch grid is not 1..E")
    raw = df["Y_raw"].to_numpy(float)
    rng = np.random.default_rng([DEQUANT_SEED, mote])
    jit = rng.uniform(-QUANTUM / 2, QUANTUM / 2, raw.size)
    y = raw + jit
    suspect = df["suspect"].to_numpy(int) == 1
    y_clean = np.where(suspect, np.nan, y)
    fs = int(epoch[np.argmax(suspect)]) if suspect.any() else None
    out = {"mote": mote, "epoch": epoch, "raw": raw, "y": y, "y_clean": y_clean,
           "suspect": suspect, "observed": np.isfinite(raw),
           "humidity": df["humidity"].to_numpy(float),
           "voltage": df["voltage"].to_numpy(float), "first_suspect": fs}
    out["climb_onset"] = climb_onset(out)
    _CACHE[key] = out
    return out


def trailing_median(epoch_obs: np.ndarray, y_obs: np.ndarray, hours: float) -> np.ndarray:
    """Median of the observed values in the trailing window (e − hours, e]."""
    lo = np.searchsorted(epoch_obs, epoch_obs - hours * EPH, side="right")
    return np.array([np.median(y_obs[a:i + 1]) for i, a in enumerate(lo)])


def climb_onset(d: dict) -> int | None:
    """Start of the out-of-range climb before the first threshold hit.

    The last observed epoch before the first suspect reading at which the
    trailing 1-h median of the raw temperature was still ≤ the maximum of the
    fit window (days 1–7, the readings the models saw). From there on the
    readings exceed a week of normal operation and climb to the threshold
    without coming back: very probably erroneous (the "climb" label).
    """
    fs = d["first_suspect"]
    if fs is None:
        return None
    ob = d["observed"]
    e_obs, y_obs = d["epoch"][ob], d["raw"][ob]
    sel = (e_obs >= fs - 3 * EPD) & (e_obs < fs)
    med = trailing_median(e_obs[sel], y_obs[sel], CLIMB_MEDIAN_H)
    top = float(np.nanmax(d["raw"][:FIT_END]))
    below = np.nonzero(med <= top)[0]
    return int(e_obs[sel][below[-1] + 1]) if below.size else int(e_obs[sel][0])


def truth_labels(d: dict, lo: int, hi: int) -> dict:
    """Row labels of epochs lo..hi (1-based, inclusive) for scoring.

    * ``suspect`` — the dataset's threshold flag (partial truth);
    * ``climb``   — [climb onset, first suspect): out-of-range climb;
    * ``ambig``   — the AMBIG_H hours before the first suspect, before the
      climb: possible early drift;
    * ``normal``  — every other observed, non-suspect row.
    """
    sl = slice(lo - 1, hi)
    e = d["epoch"][sl]
    ob = d["observed"][sl]
    s = d["suspect"][sl] & ob
    fs, co = d["first_suspect"], d["climb_onset"]
    climb = ob & ~s & (e >= co) & (e < fs)
    ambig = ob & ~s & ~climb & (e >= fs - AMBIG_H * EPH) & (e < fs)
    post = ob & ~s & (e >= fs)
    normal = ob & ~s & ~climb & ~ambig & ~post
    return {"epoch": e, "observed": ob, "suspect": s, "climb": climb, "ambig": ambig,
            "post": post, "normal": normal,
            "plateau": s & (d["raw"][sl] >= 120.0)}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def ice_cfg(kind: str, **extra) -> dict:
    """ICE settings of every fit (rs_common.estimate, best iterate returned)."""
    cfg = {"fit_margins": True, "candidates": list(rc.COPULA_CANDIDATES),
           "selection_criterion": "mle", "margin_selection_rule": rc.margin_rule(kind),
           "init": "model", "n_starts": 1, "max_iter": ICE_MAX_ITER, "tol": ICE_TOL,
           "missing_strategy": "available", "return_best_iterate": True}
    cfg.update(extra)
    return cfg


def fit_start(kind: str, K: int, Y: np.ndarray, start: int) -> tuple[PMCModel, object, str, float]:
    """One ICE run from start ``start`` (0: k-means; ≥ 1: multistart draw)."""
    init, tag = rc.starting_model(kind, K, Y, start, N_STARTS)
    t0 = time.perf_counter()
    model, trace = ice(init, Y, ice_cfg=ice_cfg(kind))
    return model, trace, tag, time.perf_counter() - t0


def exact_loglik(model: PMCModel, Y: np.ndarray) -> float:
    """log p(y_obs) of a model (forward pass with the gaps integrated out)."""
    return float(classify(model, Y)[2])


def bic(model: PMCModel, ll: float, n_obs: int) -> float:
    return float(-2.0 * ll + rc.n_free_params(model) * np.log(n_obs))


def model_path(mote: int, kind: str, K: int) -> Path:
    return MODELS_DIR / f"mote{mote}_{kind}_K{K}.toml"


def load_model(path: Path) -> PMCModel:
    import tomllib
    return PMCModel.from_dict(tomllib.loads(path.read_text()), path=str(path))


def selected_models() -> dict:
    """{(mote, kind): (K, PMCModel)} chosen by fit_clean.py (BIC)."""
    sel = json.loads((RESULTS / "selected_models.json").read_text())
    out = {}
    for key, K in sel.items():
        mote, kind = key.split("|")
        out[(int(mote), kind)] = (int(K), load_model(model_path(int(mote), kind, int(K))))
    return out


def params(model: PMCModel) -> dict:
    """States sorted by margin mean: π, mean, sd, stay probability, τ_ii."""
    K = model.K
    pi = np.asarray(model.stationary_pi, float)
    means = np.array([model.margin(i)._frozen.mean() for i in range(K)])
    sds = np.array([model.margin(i)._frozen.std() for i in range(K)])
    order = np.argsort(means)
    if model.variant.has_markov_prior:
        A = np.asarray(model.transition_A, float)
    else:
        p = np.asarray(model.prior_p, float)
        A = p / p.sum(axis=1, keepdims=True)
    out = {"order": order, "pi": pi[order], "mean": means[order], "sd": sds[order],
           "stay": np.diag(A)[order]}
    if model.variant.uses_copula:
        blocks = {(int(b["i"]), int(b["j"])): b for b in model.copula_blocks()}
        out["fam"] = [blocks[(int(i), int(i))]["name"] for i in order]
        out["tau"] = np.array([float(blocks[(int(i), int(i))].get("tau", np.nan)) for i in order])
    return out


def fmt_vec(v, f="{:.2f}") -> str:
    return " / ".join(f.format(x) for x in v)


# ---------------------------------------------------------------------------
# Baselines (on the observed rows, NaN rows skipped)
# ---------------------------------------------------------------------------

def hampel(y: np.ndarray, t: float, half: int = HAMPEL_HALF) -> np.ndarray:
    """|y_n − med| > t · 1.4826 · max(MAD, QUANTUM), centred window of observed rows."""
    from numpy.lib.stride_tricks import sliding_window_view
    y = np.asarray(y, float)
    ob = np.isfinite(y)
    yo = y[ob]
    out = np.zeros(y.size, bool)
    if yo.size < 2 * half + 1:
        return out
    pad = np.pad(yo, half, mode="reflect")
    win = sliding_window_view(pad, 2 * half + 1)
    med = np.median(win, axis=1)
    mad = 1.4826 * np.maximum(np.median(np.abs(win - med[:, None]), axis=1), QUANTUM)
    out[ob] = np.abs(yo - med) > t * mad
    return out


def robust_z(y: np.ndarray, epoch: np.ndarray) -> np.ndarray:
    """|z| of each observed row against the trailing ROBZ_WINDOW_H hours.

    z_n = (y_n − med) / (1.4826 · max(MAD, QUANTUM)) over the observed rows in
    [e_n − window, e_n); NaN when fewer than ROBZ_MIN rows (not tested).
    """
    y = np.asarray(y, float)
    ob = np.isfinite(y)
    eo, yo = epoch[ob], y[ob]
    lo = np.searchsorted(eo, eo - ROBZ_WINDOW_H * EPH, side="left")
    z = np.full(yo.size, np.nan)
    for i in range(yo.size):
        a = lo[i]
        if i - a < ROBZ_MIN:
            continue
        w = yo[a:i]
        med = np.median(w)
        mad = 1.4826 * max(np.median(np.abs(w - med)), QUANTUM)
        z[i] = abs(yo[i] - med) / mad
    out = np.full(y.size, np.nan)
    out[ob] = z
    return out


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

def episodes(flag_epochs: np.ndarray, gap_h: float = EPISODE_GAP_H) -> list[tuple[int, int, int]]:
    """(start epoch, end epoch, n flags) of the flag clusters (gaps < gap_h)."""
    fe = np.asarray(flag_epochs, int)
    if fe.size == 0:
        return []
    cut = np.nonzero(np.diff(fe) >= gap_h * EPH)[0]
    starts = np.r_[0, cut + 1]
    ends = np.r_[cut, fe.size - 1]
    return [(int(fe[a]), int(fe[b]), int(b - a + 1)) for a, b in zip(starts, ends)]


def score_flags(flag: np.ndarray, lab: dict, fs: int) -> dict:
    """Detection scores of a flag vector against the labels of truth_labels."""
    flag = np.asarray(flag, bool) & lab["observed"]
    e = lab["epoch"]
    s, cl, am, no, po = lab["suspect"], lab["climb"], lab["ambig"], lab["normal"], lab["post"]
    n_flag = int(flag.sum())
    tp = int((flag & s).sum())
    ext = s | cl
    out = {
        "n_obs": int(lab["observed"].sum()), "n_suspect": int(s.sum()), "n_flag": n_flag,
        "recall": tp / max(int(s.sum()), 1),
        "precision": tp / n_flag if n_flag else np.nan,
        "recall_rise": float((flag & s & ~lab["plateau"]).sum() / max(int((s & ~lab["plateau"]).sum()), 1)),
        "flag_climb": int((flag & cl).sum()), "n_climb": int(cl.sum()),
        "flag_ambig": int((flag & am).sum()), "n_ambig": int(am.sum()),
        "flag_normal": int((flag & no).sum()), "n_normal": int(no.sum()),
        "flag_post": int((flag & po).sum()),
        "precision_ext": float((flag & ext).sum() / n_flag) if n_flag else np.nan,
        "recall_ext": float((flag & ext).sum() / max(int(ext.sum()), 1)),
    }
    out["fa_per_1000"] = 1000.0 * out["flag_normal"] / max(out["n_normal"], 1)
    ep_normal = episodes(e[flag & no])
    out["fa_episodes"] = len(ep_normal)
    days_normal = (e[no].max() - e[no].min()) / EPD if no.any() else np.nan
    out["fa_episodes_per_day"] = len(ep_normal) / days_normal if days_normal else np.nan
    # Lead times (hours before the first threshold hit; > 0 = earlier).
    fe = e[flag]
    out["lead_first_h"] = (fs - fe[0]) / EPH if fe.size else np.nan
    pre = fe[(fe >= fs - AMBIG_H * EPH)]
    out["lead_24h_h"] = (fs - pre[0]) / EPH if pre.size else np.nan
    # The alarm episode that reaches the failure: the first one ending at most
    # EPISODE_GAP_H before the first threshold hit (negative lead = late).
    lead = np.nan
    for a, b, _ in episodes(fe):
        if b >= fs - EPISODE_GAP_H * EPH:
            lead = (fs - a) / EPH
            break
    out["lead_episode_h"] = lead
    return out


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))
