"""
reproduce_csda2013.py — Reproduce experiments from Derrode-Pieczynski (CSDA 2013).

Reproduces the three experimental sections of the paper *"Unsupervised data
classification using pairwise Markov chains with automatic copulas selection"*:

* §3.2 — PMC supervised restoration: impact of the copula shape (Tables 2, 3).
* §3.3 — Same setup with the PMM (i.i.d.) baseline (Tables 4, 5).
* §4.3 — Unsupervised ICE-based copula selection (Tables 6, 7).

Section 5 (radar image segmentation) is intentionally **not** reproduced.

Usage
-----
::

    # Quick run (~5 min on a 10-core laptop) — 30 reps for §3.2/3.3,
    # 5 ICE runs for §4.3. Auto-parallel (uses half the CPU cores).
    python report/reproduce_csda2013.py --quick

    # Full reproduction (~30-50 min) — 300 reps and 10 ICE runs as
    # in the paper.
    python report/reproduce_csda2013.py --full

    # Run only one experiment, force sequential mode for deterministic
    # log output (output CSVs are bit-identical either way).
    python report/reproduce_csda2013.py --exp1 --quick --jobs 1

    # Use all 8 cores explicitly.
    python report/reproduce_csda2013.py --full --jobs 8

Outputs
-------
``report/results/exp{1,2,3}_*.csv`` — one CSV per table.
``report/tables/exp{1,2,3}_*.tex``  — booktabs-compatible LaTeX tables.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np

from prg.copulas       import (
    CopulaA12, CopulaA14, CopulaClayton, CopulaCubSec, CopulaFGM,
    CopulaGaussian, CopulaGH, CopulaProduct, CopulaStudent,
)
from prg.pmc           import PMCModel, classify, classify_pmm, ice, simulate, simulate_pmm
from prg.pmc.inference import error_rate

logger = logging.getLogger("csda2013")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s | %(message)s")


# Hush ICE per-iteration chatter (we run hundreds of fits).
logging.getLogger("prg.pmc.ice").setLevel(logging.WARNING)
logging.getLogger("prg.pmc.model").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPORT_DIR  = Path(__file__).resolve().parent
RESULTS_DIR = REPORT_DIR / "results"
TABLES_DIR  = REPORT_DIR / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Paper notation ↔ package classes
# ---------------------------------------------------------------------------
# Code names (cN) match those of CSDA-2013 Table A.11.
COPULA_REGISTRY: dict[str, dict] = {
    "c0": {"name": "Product",  "klass": CopulaProduct,  "short": "Prod",    "tau_range": (0.0, 0.0)},
    "c1": {"name": "Gauss",    "klass": CopulaGaussian, "short": "Gauss",   "tau_range": (-1.0, 1.0)},
    "c2": {"name": "Student",  "klass": CopulaStudent,  "short": "Student", "tau_range": (-1.0, 1.0)},
    "c3": {"name": "GH",       "klass": CopulaGH,       "short": "GH",      "tau_range": (0.0, 1.0)},
    "c4": {"name": "FGM",      "klass": CopulaFGM,      "short": "FGM",     "tau_range": (-2/9, 2/9)},
    "c5": {"name": "CubSec",   "klass": CopulaCubSec,   "short": "CubSec",  "tau_range": (0.0, 33/200)},
    "c6": {"name": "Clayton",  "klass": CopulaClayton,  "short": "Clayton", "tau_range": (0.0, 1.0)},
    "c7": {"name": "A12",      "klass": CopulaA12,      "short": "A12",     "tau_range": (1/3, 1.0)},
    "c8": {"name": "A14",      "klass": CopulaA14,      "short": "A14",     "tau_range": (1/3, 1.0)},
}

# Eligible copula sets used by the paper.
PI_LOW  = ["c0", "c1", "c2", "c3", "c4", "c5", "c6"]    # τ#1 = 0.16
PI_HIGH = ["c0", "c1", "c2", "c3", "c6", "c7", "c8"]    # τ#2 = 0.70


# ---------------------------------------------------------------------------
# Margin parameters (CSDA Table 1)
# ---------------------------------------------------------------------------

# Gaussian margins f_{ij} (paper notation; under SR-PMC tying f_{ij} = f_i,
# so we collapse to one Gaussian per state taking the j=0 anchor: f_{i0}).
# This matches the K-format used by our PMCModel parser.
GAUSSIAN_MARGINS_K2: list[dict] = [
    {"i": 0, "dist": "norm", "params": {"loc": 0.0, "scale": 1.0}},   # f_0 = f_{00}
    {"i": 1, "dist": "norm", "params": {"loc": 1.1, "scale": 1.4}},   # f_1 = f_{10}
]

# Gamma margins approximated to match the Gaussian moments
# (paper claims same std as the Gaussians — Table 1). With shape α fixed
# we derive scale = σ/√α and loc = μ − α·scale.
def _gamma_match_moments(mu: float, sigma: float, alpha: float) -> dict:
    """Return scipy.stats `gamma` params giving mean μ and std σ with shape α."""
    scale = sigma / math.sqrt(alpha)
    loc   = mu - alpha * scale
    return {"a": alpha, "loc": loc, "scale": scale}

GAMMA_MARGINS_K2: list[dict] = [
    {"i": 0, "dist": "gamma", "params": _gamma_match_moments(0.0, 1.0, 8.0)},
    {"i": 1, "dist": "gamma", "params": _gamma_match_moments(1.1, 1.4, 3.0)},
]


# ---------------------------------------------------------------------------
# Helpers — model factory
# ---------------------------------------------------------------------------

def _build_pmc_model(
    copula_code: str,
    tau:         float,
    margins:     list[dict],
    p:           list[list[float]],
    name:        str = "exp",
) -> PMCModel:
    """Build an SR-PMC model with the same copula on all 4 transitions."""
    info = COPULA_REGISTRY[copula_code]
    raw = {
        "model":  {"name": name, "variant": "PMC", "K": 2, "N_default": 2000},
        "prior":  {"p": p},
        "margins": margins,
        "copulas": [
            {"i": i, "j": j, "name": info["short"], "tau": float(tau)}
            for i in range(2) for j in range(2)
        ],
    }
    # Student copula needs its df extra parameter.
    if copula_code == "c2":
        for blk in raw["copulas"]:
            blk["df"] = 4.0
    return PMCModel.from_dict(raw)


def _clip_tau_to_range(tau: float, copula_code: str) -> float:
    """Return tau clipped into the family's admissible range (with EPS pad).

    For degenerate ranges where lo == hi (Product copula has τ ≡ 0) we
    return the unique admissible value directly — no padding.
    """
    lo, hi = COPULA_REGISTRY[copula_code]["tau_range"]
    span   = hi - lo
    if span <= 1e-9:
        return float(lo)
    pad = 1e-3 * span
    return float(np.clip(tau, lo + pad, hi - pad))


# ---------------------------------------------------------------------------
# Multiprocessing helpers
# ---------------------------------------------------------------------------
#
# Reps and ICE runs in this script are independent and embarrassingly
# parallel. We dispatch them via :class:`ProcessPoolExecutor` so a 10-core
# laptop drops the ``--full`` wall time from ~3 h to ~30-40 min using
# half the cores (the user-requested default).
#
# Workers re-build the relevant ``PMCModel`` instances on each call from
# tiny picklable arguments (margins/prior dicts, sim_code strings,
# scalar τ values). Model construction is ~1 ms — negligible vs. the
# ~300 ms classification work or the ~2 s ICE run, so we keep the
# implementation simple and avoid per-process model caching.


# ---------------------------------------------------------------------------
# Progress / ETA helper (stdlib-only; no tqdm dep)
# ---------------------------------------------------------------------------

T = TypeVar("T")


def _fmt_dur(secs: float) -> str:
    """Compact duration string: ``42s`` / ``3m07s`` / ``1h12m``."""
    secs = max(0, int(secs))
    if secs < 60:
        return f"{secs}s"
    m, s = divmod(secs, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _is_tty(stream) -> bool:
    try:
        return stream.isatty()
    except (AttributeError, OSError):                     # pragma: no cover
        return False


def _progress(
    iterable: Iterable[T],
    *,
    total:        int,
    desc:         str = "",
    stream        = None,
    min_interval: float = 0.2,
) -> Iterator[T]:
    """Wrap ``iterable`` with a self-overwriting progress bar + ETA.

    Yields each item from the source unchanged, printing the running
    ``N/total`` count, percentage, elapsed and ETA on every step but
    rate-limiting the *display* refreshes to ``min_interval`` seconds
    so the cost is negligible regardless of inner-loop speed.

    On a non-TTY stream (e.g. piped to a log file) we skip the ``\\r``
    rewrite and emit one summary line per ~10 % so the file stays
    readable.

    Designed as a drop-in for ``executor.map(...)`` returns: total is
    given explicitly because the iterator has no ``__len__``.
    """
    if stream is None:
        stream = sys.stderr
    is_tty   = _is_tty(stream)
    width    = 24
    t0       = time.perf_counter()
    last_t   = -1.0
    log_step = max(1, total // 10)              # for non-TTY emission
    log_min_dt = 3.0                            # non-TTY min seconds between lines

    for k, item in enumerate(iterable, 1):
        yield item
        now      = time.perf_counter()
        is_last  = (k == total)
        if not is_last:
            if is_tty:
                # TTY: throttle by wall time only — \r overwriting is cheap.
                if (now - last_t) < min_interval:
                    continue
            else:
                # Non-TTY (piped to a log file): emit at most once per ~10 %
                # of total AND at most once every ``log_min_dt`` seconds, so
                # the log stays both informative and uncluttered.
                if (k % log_step) != 0:
                    continue
                if (now - last_t) < log_min_dt:
                    continue
        last_t = now
        elapsed = now - t0
        rate    = k / max(elapsed, 1e-9)
        eta     = (total - k) / max(rate, 1e-9)
        filled  = int(round(width * k / total))
        bar     = "█" * filled + "·" * (width - filled)
        line    = (f"{desc}  [{bar}]  {k:>4}/{total}  ({100*k/total:5.1f}%)"
                   f"  elapsed={_fmt_dur(elapsed)}"
                   f"  ETA={'   —  ' if is_last else _fmt_dur(eta)}")
        if is_tty:
            # \r + clear-to-end-of-line so a shorter line never leaves
            # stale characters from the previous one.
            stream.write("\r\033[K" + line)
            if is_last:
                stream.write("\n")
        else:
            stream.write(line + "\n")
        stream.flush()


def _default_jobs() -> int:
    """Half the available CPU cores, floored at 1 — the user-requested default.

    Picks up ``os.sched_getaffinity`` when available (Linux) so we play
    nicely with cgroups / taskset; falls back to ``os.cpu_count``.
    """
    n = None
    if hasattr(os, "sched_getaffinity"):
        try:
            n = len(os.sched_getaffinity(0))
        except OSError:                                   # pragma: no cover
            n = None
    if n is None:
        n = os.cpu_count() or 1
    return max(1, n // 2)


def _exp_pmc_rep_worker(args: tuple) -> np.ndarray:
    """One PMC supervised rep — pickle-safe, used by exp1.

    ``args`` carries everything needed to rebuild the models locally;
    no PMCModel objects cross the process boundary (each is ~2 MB
    pickled, so per-task pickling would dominate).
    """
    sim_code, est_codes, tau, margins, prior, n_obs, seed = args
    sim_mdl = _build_pmc_model(
        sim_code, _clip_tau_to_range(tau, sim_code), margins, prior,
    )
    est_mdls = [
        _build_pmc_model(c, _clip_tau_to_range(tau, c), margins, prior)
        for c in est_codes
    ]
    X_ref, Y = simulate(sim_mdl, N=n_obs, seed=seed)
    return np.array(
        [error_rate(X_ref, classify(m, Y)[0]) for m in est_mdls],
        dtype=float,
    )


def _exp_pmm_rep_worker(args: tuple) -> np.ndarray:
    """One PMM (i.i.d.) rep — pickle-safe, used by exp2."""
    sim_code, est_codes, tau, margins, prior, n_obs, seed = args
    sim_mdl = _build_pmc_model(
        sim_code, _clip_tau_to_range(tau, sim_code), margins, prior,
    )
    est_mdls = [
        _build_pmc_model(c, _clip_tau_to_range(tau, c), margins, prior)
        for c in est_codes
    ]
    n_pairs = n_obs // 2
    X_ref, Y = simulate_pmm(sim_mdl, n_pairs=n_pairs, seed=seed)
    return np.array(
        [error_rate(X_ref, classify_pmm(m, Y)[0]) for m in est_mdls],
        dtype=float,
    )


def _exp3_run_worker(args: tuple) -> dict:
    """One ICE run for exp3 — returns the per-run summary dict.

    The selected family + Kendall τ are returned per pair (i, j) so
    the parent can tally hit-counts and mean-τ identically to the
    sequential code path.
    """
    (truth_layout, candidates_short, init_layout, margins, prior, n_obs,
     max_iter, criterion, seed) = args
    truth_mdl = _build_pmc_model_per_pair(truth_layout, margins, prior)
    init_mdl  = _build_pmc_model_per_pair(init_layout,  margins, prior)
    X_ref, Y = simulate(truth_mdl, N=n_obs, seed=seed)
    fitted, _trace = ice(init_mdl, Y, ice_cfg={
        "max_iter":           max_iter,
        "candidates":         candidates_short,
        "fit_margins":        False,
        "selection_criterion": criterion,
    })
    K = truth_mdl.K
    selected = {}
    for i in range(K):
        for j in range(K):
            cop = fitted.copula(i, j)
            selected[(i, j)] = (
                cop.copula_enum.value.SHORT_NAME,
                float(cop.params["tau_k"]),
            )
    X_hat_sup,   _, _ = classify(truth_mdl, Y)
    X_hat_unsup, _, _ = classify(fitted,    Y)
    return {
        "selected":  selected,
        "sup_err":   float(error_rate(X_ref, X_hat_sup)),
        "unsup_err": float(error_rate(X_ref, X_hat_unsup)),
    }


# ---------------------------------------------------------------------------
# Experiment #1 — PMC supervised, impact of copula shape
# ---------------------------------------------------------------------------

# Joint prior matching CSDA §3.2 (asymmetric but symmetric for SR-PMC).
PMC_PRIOR_ORIG = [[0.50, 0.05], [0.05, 0.40]]


@dataclass
class _ExpConfig:
    label:        str
    tau:          float
    copulas:      list[str]
    margin_kind:  str                   # "gauss" or "gamma"
    margins:      list[dict]
    n_obs:        int
    reps:         int


def _run_pmc_supervised_row(
    sim_code:   str,
    cfg:        _ExpConfig,
    seed_base:  int,
    *,
    executor:   ProcessPoolExecutor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one *row* of the table.

    For a single simulation copula ``sim_code``, generates ``cfg.reps``
    PMC sequences and classifies each of them with every estimator in
    ``cfg.copulas``. This is faithful to CSDA-2013's protocol (the
    paper writes "we restored simulated observations [...] using all
    simulation parameters except the copula shape"), and it is also the
    only computationally tractable strategy when the simulation copula
    is itself slow (Student, GH, A12, A14 — all rely on ``brentq``-based
    Rosenblatt inversion).

    When ``executor`` is given, reps are dispatched in parallel; results
    are aggregated in a stable order so the final ``(means, stds)`` are
    bit-identical to the sequential path for the same seeds.

    Returns ``(means, stds)``, both 1-D arrays of length ``len(cfg.copulas)``
    in the same order as ``cfg.copulas``, expressed as percentages.
    """
    n_est = len(cfg.copulas)
    args_list = [
        (sim_code, list(cfg.copulas), cfg.tau, cfg.margins, PMC_PRIOR_ORIG,
         cfg.n_obs, seed_base + r)
        for r in range(cfg.reps)
    ]
    err = np.empty((cfg.reps, n_est), dtype=float)
    if executor is None:
        source: Iterable[np.ndarray] = (
            _exp_pmc_rep_worker(a) for a in args_list
        )
    else:
        # Preserve order (chunksize tuned for ~30-300 reps; small enough
        # for load-balancing, large enough to amortise pickle overhead).
        source = executor.map(_exp_pmc_rep_worker, args_list, chunksize=4)
    desc = f"  exp1 / {cfg.label} / row {sim_code:>3s}"
    for r, vec in enumerate(_progress(source, total=cfg.reps, desc=desc)):
        err[r, :] = vec
    return 100.0 * err.mean(axis=0), 100.0 * err.std(axis=0)


def run_exp1(
    reps:      int,
    n_obs:     int = 2000,
    *,
    executor:  ProcessPoolExecutor | None = None,
) -> dict[str, dict]:
    """Run §3.2 — Tables 2 and 3 of CSDA 2013."""
    configs = [
        _ExpConfig("low_tau_gauss",  tau=0.16, copulas=PI_LOW,
                   margin_kind="gauss", margins=GAUSSIAN_MARGINS_K2,
                   n_obs=n_obs, reps=reps),
        _ExpConfig("low_tau_gamma",  tau=0.16, copulas=PI_LOW,
                   margin_kind="gamma", margins=GAMMA_MARGINS_K2,
                   n_obs=n_obs, reps=reps),
        _ExpConfig("high_tau_gauss", tau=0.70, copulas=PI_HIGH,
                   margin_kind="gauss", margins=GAUSSIAN_MARGINS_K2,
                   n_obs=n_obs, reps=reps),
        _ExpConfig("high_tau_gamma", tau=0.70, copulas=PI_HIGH,
                   margin_kind="gamma", margins=GAMMA_MARGINS_K2,
                   n_obs=n_obs, reps=reps),
    ]
    results: dict[str, dict] = {}
    for cfg in configs:
        logger.info(
            "Exp #1 / %s : τ=%.2f  Π=%s  margins=%s  reps=%d  N=%d",
            cfg.label, cfg.tau, cfg.copulas, cfg.margin_kind, cfg.reps, cfg.n_obs,
        )
        means = np.full((len(cfg.copulas), len(cfg.copulas)), np.nan)
        stds  = np.full_like(means, np.nan)
        t0 = time.time()
        for r, sim_code in enumerate(cfg.copulas):
            row_means, row_stds = _run_pmc_supervised_row(
                sim_code, cfg, seed_base=1000 + r * 100,
                executor=executor,
            )
            means[r, :] = row_means
            stds[r,  :] = row_stds
            logger.info(
                "  row %s  [%2d/%2d]  elapsed=%.1fs",
                sim_code, r + 1, len(cfg.copulas), time.time() - t0,
            )
        results[cfg.label] = {
            "copulas": list(cfg.copulas),
            "means":   means,
            "stds":    stds,
            "tau":     cfg.tau,
            "margin":  cfg.margin_kind,
            "reps":    cfg.reps,
            "n_obs":   cfg.n_obs,
        }
        _save_table_csv(RESULTS_DIR / f"exp1_{cfg.label}.csv",
                        cfg.copulas, means, stds, header_note=(
                            f"Exp #1 — PMC supervised, τ={cfg.tau}, "
                            f"margins={cfg.margin_kind}, reps={cfg.reps}, N={cfg.n_obs}"
                        ))
    return results


# ---------------------------------------------------------------------------
# Experiment #2 — PMM (i.i.d.) baseline
# ---------------------------------------------------------------------------

def _run_pmm_supervised_row(
    sim_code: str,
    cfg:      _ExpConfig,
    seed_base: int,
    *,
    executor: ProcessPoolExecutor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """PMM analogue of :func:`_run_pmc_supervised_row`."""
    n_est = len(cfg.copulas)
    args_list = [
        (sim_code, list(cfg.copulas), cfg.tau, cfg.margins, PMC_PRIOR_ORIG,
         cfg.n_obs, seed_base + r)
        for r in range(cfg.reps)
    ]
    err = np.empty((cfg.reps, n_est), dtype=float)
    if executor is None:
        source: Iterable[np.ndarray] = (
            _exp_pmm_rep_worker(a) for a in args_list
        )
    else:
        source = executor.map(_exp_pmm_rep_worker, args_list, chunksize=4)
    desc = f"  exp2 / {cfg.label} / row {sim_code:>3s}"
    for r, vec in enumerate(_progress(source, total=cfg.reps, desc=desc)):
        err[r, :] = vec
    return 100.0 * err.mean(axis=0), 100.0 * err.std(axis=0)


def run_exp2(
    reps:      int,
    n_obs:     int = 2000,
    *,
    executor:  ProcessPoolExecutor | None = None,
) -> dict[str, dict]:
    """Run §3.3 — Tables 4 and 5 (PMM)."""
    configs = [
        _ExpConfig("low_tau_gauss",  tau=0.16, copulas=PI_LOW,
                   margin_kind="gauss", margins=GAUSSIAN_MARGINS_K2,
                   n_obs=n_obs, reps=reps),
        _ExpConfig("low_tau_gamma",  tau=0.16, copulas=PI_LOW,
                   margin_kind="gamma", margins=GAMMA_MARGINS_K2,
                   n_obs=n_obs, reps=reps),
        _ExpConfig("high_tau_gauss", tau=0.70, copulas=PI_HIGH,
                   margin_kind="gauss", margins=GAUSSIAN_MARGINS_K2,
                   n_obs=n_obs, reps=reps),
        _ExpConfig("high_tau_gamma", tau=0.70, copulas=PI_HIGH,
                   margin_kind="gamma", margins=GAMMA_MARGINS_K2,
                   n_obs=n_obs, reps=reps),
    ]
    results: dict[str, dict] = {}
    for cfg in configs:
        logger.info(
            "Exp #2 / %s : τ=%.2f  Π=%s  margins=%s  reps=%d",
            cfg.label, cfg.tau, cfg.copulas, cfg.margin_kind, cfg.reps,
        )
        means = np.full((len(cfg.copulas), len(cfg.copulas)), np.nan)
        stds  = np.full_like(means, np.nan)
        t0 = time.time()
        for r, sim_code in enumerate(cfg.copulas):
            row_means, row_stds = _run_pmm_supervised_row(
                sim_code, cfg, seed_base=2000 + r * 100,
                executor=executor,
            )
            means[r, :] = row_means
            stds[r,  :] = row_stds
            logger.info(
                "  row %s  [%2d/%2d]  elapsed=%.1fs",
                sim_code, r + 1, len(cfg.copulas), time.time() - t0,
            )
        results[cfg.label] = {
            "copulas": list(cfg.copulas), "means": means, "stds": stds,
            "tau": cfg.tau, "margin": cfg.margin_kind, "reps": cfg.reps,
            "n_obs": cfg.n_obs,
        }
        _save_table_csv(RESULTS_DIR / f"exp2_{cfg.label}.csv",
                        cfg.copulas, means, stds, header_note=(
                            f"Exp #2 — PMM (i.i.d.), τ={cfg.tau}, "
                            f"margins={cfg.margin_kind}, reps={cfg.reps}, N={cfg.n_obs}"
                        ))
    return results


# ---------------------------------------------------------------------------
# Experiment #3 — ICE-based copula selection
# ---------------------------------------------------------------------------

@dataclass
class _IceExpConfig:
    label:    str
    candidates: list[str]
    truth:    dict[tuple[int, int], tuple[str, float]]  # (i,j) → (code, tau)
    margins:  list[dict]
    margin_kind: str
    p:        list[list[float]]
    runs:     int
    n_obs:    int = 2500
    max_iter: int = 30


def _build_pmc_model_per_pair(
    truth:   dict[tuple[int, int], tuple[str, float]],
    margins: list[dict],
    p:       list[list[float]],
) -> PMCModel:
    """Build a PMC model with potentially different copulas per (i, j)."""
    K = 2
    copulas = []
    for (i, j), (code, tau) in truth.items():
        info = COPULA_REGISTRY[code]
        blk  = {"i": i, "j": j, "name": info["short"],
                "tau": _clip_tau_to_range(tau, code)}
        if code == "c2":
            blk["df"] = 4.0
        copulas.append(blk)
    raw = {
        "model":  {"name": "exp3", "variant": "PMC", "K": K, "N_default": 2500},
        "prior":  {"p": p},
        "margins": margins,
        "copulas": copulas,
    }
    return PMCModel.from_dict(raw)


def run_exp3(
    runs: int,
    criteria: list[str] | None = None,
    *,
    executor: ProcessPoolExecutor | None = None,
) -> dict[str, dict]:
    """Run §4.3 — Tables 6 and 7 (ICE-based copula selection).

    Parameters
    ----------
    runs     : number of independent simulations.
    criteria : list of selection criteria to evaluate. Default is the
               5-element list ``["mle", "aic", "bic", "huard", "cvm"]``.
               Each criterion is run on each config; results are keyed
               ``f"{cfg.label}__{criterion}"`` so the LaTeX writer can
               group them naturally.
    executor : optional :class:`ProcessPoolExecutor` for parallel ICE
               runs (the dominant cost for ``--full``).
    """
    if criteria is None:
        criteria = ["mle", "aic", "bic", "huard", "cvm"]

    p_orig = [[0.50, 0.05], [0.05, 0.40]]
    p_bal  = [[0.35, 0.15], [0.15, 0.35]]

    base_configs = [
        _IceExpConfig(
            label="exp1_orig_p",
            candidates=["c1", "c3", "c6"],   # Gauss, GH, Clayton
            truth={(0, 0): ("c1", 0.7), (0, 1): ("c3", 0.4),
                   (1, 0): ("c3", 0.4), (1, 1): ("c6", 0.7)},
            margins=GAUSSIAN_MARGINS_K2, margin_kind="gauss",
            p=p_orig, runs=runs,
        ),
        _IceExpConfig(
            label="exp2_orig_p",
            candidates=["c2", "c3", "c5", "c6"],   # Student, GH, CubSec, Clayton
            truth={(0, 0): ("c2", 0.25), (0, 1): ("c5", 0.10),
                   (1, 0): ("c5", 0.10), (1, 1): ("c6", 0.20)},
            margins=GAMMA_MARGINS_K2, margin_kind="gamma",
            p=p_orig, runs=runs,
        ),
        _IceExpConfig(
            label="exp1_bal_p",
            candidates=["c1", "c3", "c6"],
            truth={(0, 0): ("c1", 0.7), (0, 1): ("c3", 0.4),
                   (1, 0): ("c3", 0.4), (1, 1): ("c6", 0.7)},
            margins=GAUSSIAN_MARGINS_K2, margin_kind="gauss",
            p=p_bal, runs=runs,
        ),
        _IceExpConfig(
            label="exp2_bal_p",
            candidates=["c2", "c3", "c5", "c6"],
            truth={(0, 0): ("c2", 0.25), (0, 1): ("c5", 0.10),
                   (1, 0): ("c5", 0.10), (1, 1): ("c6", 0.20)},
            margins=GAMMA_MARGINS_K2, margin_kind="gamma",
            p=p_bal, runs=runs,
        ),
    ]
    # Cross-product (config × criterion).
    configs: list[tuple[_IceExpConfig, str]] = []
    for base in base_configs:
        for crit in criteria:
            configs.append((base, crit))

    results: dict[str, dict] = {}
    for base_cfg, crit in configs:
        # Tag the cfg with the criterion (logging only); the actual
        # criterion is passed via ice_cfg below.
        cfg = base_cfg
        # Override label to include the criterion for distinct CSV/TeX paths.
        cfg = type(cfg)(**{**cfg.__dict__, "label": f"{base_cfg.label}__{crit}"})
        logger.info(
            "Exp #3 / %s : truth=%s candidates=%s runs=%d N=%d",
            cfg.label, {k: (v[0], v[1]) for k, v in cfg.truth.items()},
            cfg.candidates, cfg.runs, cfg.n_obs,
        )
        # Counters per (i, j)
        K = 2
        truth_short = {
            (i, j): COPULA_REGISTRY[cfg.truth[(i, j)][0]]["short"]
            for i in range(K) for j in range(K)
        }
        true_tau = {
            (i, j): cfg.truth[(i, j)][1] for i in range(K) for j in range(K)
        }
        hit_count = {(i, j): 0 for i in range(K) for j in range(K)}
        sum_tau   = {(i, j): 0.0 for i in range(K) for j in range(K)}
        # Error rates
        sup_err = np.empty(cfg.runs, dtype=float)
        unsup_err = np.empty(cfg.runs, dtype=float)
        # Candidate SHORT_NAMEs for ICE. The truth/init models themselves
        # are rebuilt per-run inside the worker so no PMCModel object
        # needs to cross the process boundary.
        cand_short = [COPULA_REGISTRY[c]["short"] for c in cfg.candidates]

        # Initial model layout — CSDA's kmeans-style heuristic: start
        # every (i, j) on the first candidate at τ = 0.
        init_layout = {
            (i, j): (cfg.candidates[0], 0.0) for i in range(K) for j in range(K)
        }
        # Per-run argument tuple — picklable, picklable, picklable (no
        # PMCModel crosses the process boundary).
        run_args = [
            (cfg.truth, cand_short, init_layout, cfg.margins, cfg.p,
             cfg.n_obs, cfg.max_iter, crit, 3000 + r)
            for r in range(cfg.runs)
        ]

        if executor is None:
            source: Iterable[dict] = (_exp3_run_worker(a) for a in run_args)
        else:
            # ICE runs are seconds-long, so chunksize=1 is fine for
            # load-balancing without significant pickle overhead.
            source = executor.map(_exp3_run_worker, run_args, chunksize=1)
        desc = f"  exp3 / {cfg.label}"

        t0 = time.time()
        # Tally hits, mean τ, and error rates as runs come back. Streaming
        # rather than collecting first lets the progress bar and the tally
        # advance in lockstep with worker completions.
        for r, res in enumerate(_progress(source, total=cfg.runs, desc=desc)):
            for (i, j), (selected, tau_k) in res["selected"].items():
                if selected == truth_short[(i, j)]:
                    hit_count[(i, j)] += 1
                sum_tau[(i, j)] += tau_k
            sup_err[r]   = res["sup_err"]
            unsup_err[r] = res["unsup_err"]
        logger.info("  %d/%d ICE runs done  elapsed=%.1fs",
                    cfg.runs, cfg.runs, time.time() - t0)

        # Build the (i, j) report rows.
        rows = []
        for i in range(K):
            for j in range(K):
                rows.append({
                    "i": i, "j": j,
                    "true_name":  truth_short[(i, j)],
                    "true_tau":   true_tau[(i, j)],
                    "hits":       hit_count[(i, j)],
                    "runs":       cfg.runs,
                    "mean_tau":   sum_tau[(i, j)] / cfg.runs,
                })
        results[cfg.label] = {
            "rows":         rows,
            "candidates":   cfg.candidates,
            "p":            cfg.p,
            "margin":       cfg.margin_kind,
            "runs":         cfg.runs,
            "n_obs":        cfg.n_obs,
            "sup_err_mean": float(100.0 * sup_err.mean()),
            "sup_err_std":  float(100.0 * sup_err.std()),
            "unsup_err_mean": float(100.0 * unsup_err.mean()),
            "unsup_err_std":  float(100.0 * unsup_err.std()),
        }
        # Save CSV
        out = RESULTS_DIR / f"exp3_{cfg.label}.csv"
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["i", "j", "true_name", "true_tau", "hits/runs", "mean_tau"])
            for row in rows:
                w.writerow([row["i"], row["j"], row["true_name"],
                            f"{row['true_tau']:.3f}",
                            f"{row['hits']}/{row['runs']}",
                            f"{row['mean_tau']:.3f}"])
            w.writerow([])
            w.writerow(["sup_err_mean (%)",   f"{results[cfg.label]['sup_err_mean']:.2f}"])
            w.writerow(["sup_err_std (%)",    f"{results[cfg.label]['sup_err_std']:.2f}"])
            w.writerow(["unsup_err_mean (%)", f"{results[cfg.label]['unsup_err_mean']:.2f}"])
            w.writerow(["unsup_err_std (%)",  f"{results[cfg.label]['unsup_err_std']:.2f}"])

    return results


# ---------------------------------------------------------------------------
# CSV / LaTeX writers
# ---------------------------------------------------------------------------

def _save_table_csv(
    path:        Path,
    copulas:     Iterable[str],
    means:       np.ndarray,
    stds:        np.ndarray,
    header_note: str = "",
):
    copulas = list(copulas)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        if header_note:
            w.writerow([f"# {header_note}"])
        w.writerow(["sim/est"] + copulas)
        for r, cop_r in enumerate(copulas):
            row = [cop_r]
            for c in range(len(copulas)):
                row.append(f"{means[r, c]:.2f} ({stds[r, c]:.1f})")
            w.writerow(row)


def write_latex_table_grid(
    path:    Path,
    label:   str,
    caption: str,
    copulas: Iterable[str],
    means:   np.ndarray,
    stds:    np.ndarray,
    bold_min_per_column: bool = True,
):
    """Write a 7×7 booktabs table; underlines/bolds the minimum of each column."""
    copulas = list(copulas)
    K = len(copulas)
    cols_spec = "l" + "c" * K
    lines = [
        r"\begin{table}[htbp]", r"  \centering",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        rf"  \begin{{tabular}}{{{cols_spec}}}",
        r"    \toprule",
    ]
    header = "    sim $\\backslash$ est & " + " & ".join(
        rf"$\mathbf{{c^{{{c[1:]}}}}}$" for c in copulas
    ) + r" \\"
    lines.append(header)
    lines.append(r"    \midrule")
    # Find minimum per column for highlighting.
    if bold_min_per_column and means.size:
        col_mins = [int(np.argmin(means[:, c])) for c in range(K)]
    else:
        col_mins = [-1] * K
    for r, cop_r in enumerate(copulas):
        cells = [rf"$c^{{{cop_r[1:]}}}$"]
        for c in range(K):
            txt = f"{means[r, c]:.2f} ({stds[r, c]:.1f})"
            if r == col_mins[c]:
                txt = rf"\textbf{{\underline{{{means[r, c]:.2f}}}}} ({stds[r, c]:.1f})"
            cells.append(txt)
        lines.append("    " + " & ".join(cells) + r" \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n")


def write_latex_table_ice(
    path:        Path,
    label:       str,
    caption:     str,
    rows:        list[dict],
    sup_err:     tuple[float, float],
    unsup_err:   tuple[float, float],
):
    """Write the §4.3 small table (4 rows × hit-count + mean-tau columns)."""
    lines = [
        r"\begin{table}[htbp]", r"  \centering",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        r"  \begin{tabular}{ccccc}",
        r"    \toprule",
        r"    $(i, j)$ & true family & $\tau_{ij}$ & hits / runs & $\hat\tau_{ij}$ \\",
        r"    \midrule",
    ]
    for row in rows:
        lines.append(
            f"    ({row['i']}, {row['j']}) & "
            rf"\texttt{{{row['true_name']}}} & "
            f"{row['true_tau']:.2f} & "
            f"{row['hits']}/{row['runs']} & "
            f"{row['mean_tau']:.3f} \\\\"
        )
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(rf"  \par\smallskip Supervised error: {sup_err[0]:.2f}\% "
                 rf"($\pm$ {sup_err[1]:.2f}). "
                 rf"Unsupervised: {unsup_err[0]:.2f}\% "
                 rf"($\pm$ {unsup_err[1]:.2f}).")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n")


def write_all_latex_tables(
    res1: dict | None,
    res2: dict | None,
    res3: dict | None,
):
    """Generate all .tex files from the in-memory results dicts."""
    captions_e1 = {
        "low_tau_gauss":  r"PMC supervised — $\tau = 0.16$, Gaussian margins.",
        "low_tau_gamma":  r"PMC supervised — $\tau = 0.16$, Gamma margins.",
        "high_tau_gauss": r"PMC supervised — $\tau = 0.70$, Gaussian margins.",
        "high_tau_gamma": r"PMC supervised — $\tau = 0.70$, Gamma margins.",
    }
    captions_e2 = {k: v.replace("PMC", "PMM (i.i.d.)") for k, v in captions_e1.items()}

    if res1:
        for k, r in res1.items():
            write_latex_table_grid(
                TABLES_DIR / f"exp1_{k}.tex",
                label=f"tab:exp1-{k}",
                caption=captions_e1[k],
                copulas=r["copulas"], means=r["means"], stds=r["stds"],
            )
    if res2:
        for k, r in res2.items():
            write_latex_table_grid(
                TABLES_DIR / f"exp2_{k}.tex",
                label=f"tab:exp2-{k}",
                caption=captions_e2[k],
                copulas=r["copulas"], means=r["means"], stds=r["stds"],
            )
    if res3:
        cfg_captions = {
            "exp1_orig_p":
                r"exp.\ 1 (\{Gauss, GH, Clayton\}, $p_{\text{orig}}$)",
            "exp2_orig_p":
                r"exp.\ 2 (\{Student, GH, CubSec, Clayton\}, $p_{\text{orig}}$)",
            "exp1_bal_p":
                r"exp.\ 1 with balanced prior $p_{\text{bal}}$",
            "exp2_bal_p":
                r"exp.\ 2 with balanced prior $p_{\text{bal}}$",
        }
        crit_labels = {
            "mle":   "MLE (max log-likelihood)",
            "aic":   "AIC",
            "bic":   "BIC",
            "huard": r"Huard et al.\ 2006 (CSDA-2013 Eq.\ 20)",
            "cvm":   r"Cram\'er--von Mises",
        }
        for k, r in res3.items():
            # Keys are "<cfg_label>__<criterion>" or just "<cfg_label>"
            # for the legacy single-criterion call.
            if "__" in k:
                cfg_part, crit_part = k.split("__", 1)
                crit_descr = crit_labels.get(crit_part, crit_part)
            else:
                cfg_part   = k
                crit_descr = crit_labels.get("mle", "MLE")
            cfg_descr = cfg_captions.get(cfg_part, cfg_part)
            caption = (
                rf"ICE copula selection — {cfg_descr}, criterion: {crit_descr}."
            )
            write_latex_table_ice(
                TABLES_DIR / f"exp3_{k}.tex",
                label=f"tab:exp3-{k}",
                caption=caption,
                rows=r["rows"],
                sup_err=(r["sup_err_mean"],   r["sup_err_std"]),
                unsup_err=(r["unsup_err_mean"], r["unsup_err_std"]),
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _save_run_metadata(
    reps:    int,
    runs:    int,
    n_obs:   int,
    mode:    str,
    exp1:    bool,
    exp2:    bool,
    exp3:    bool,
) -> None:
    """Write ``tables/run_metadata.tex`` defining LaTeX macros that the
    main report consumes via ``\\input``. Keeps the prose of
    ``csda2013_reproduction.tex`` consistent with the actual run
    parameters (number of replications, ICE runs, sequence length,
    mode label, date, list of executed experiments).

    Macros produced
    ---------------
    * ``\\runReps``      — number of replications per cell (exp1/exp2)
    * ``\\runIceRuns``   — number of ICE runs (exp3)
    * ``\\runNobs``      — sequence length (exp1/exp2)
    * ``\\runMode``      — ``quick`` / ``full`` / ``custom``
    * ``\\runDate``      — date the script was executed
    * ``\\runExpsList``  — comma-separated list of experiments included
                          in the current run
    """
    import datetime as _dt

    parts = []
    if exp1:
        parts.append("\\S3.2")
    if exp2:
        parts.append("\\S3.3")
    if exp3:
        parts.append("\\S4.3")
    exps_list = ", ".join(parts) if parts else "none"

    # Use f-strings throughout — embedding "%" comments inside a "%"-format
    # template would collide with Python's printf-style operator.
    today = _dt.date.today().isoformat()
    lines = [
        "% Auto-generated by report/reproduce_csda2013.py — do not edit.",
        "% Defines LaTeX macros consumed by the report's prose so that",
        "% reps / runs / mode stay consistent across --quick and --full.",
        f"\\providecommand{{\\runReps}}{{{reps}}}",
        f"\\providecommand{{\\runIceRuns}}{{{runs}}}",
        f"\\providecommand{{\\runNobs}}{{{n_obs}}}",
        f"\\providecommand{{\\runMode}}{{{mode}}}",
        f"\\providecommand{{\\runDate}}{{{today}}}",
        f"\\providecommand{{\\runExpsList}}{{{exps_list}}}",
        "",
    ]
    out = TABLES_DIR / "run_metadata.tex"
    out.write_text("\n".join(lines))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce CSDA 2013 experiments (§3.2, §3.3, §4.3)."
    )
    parser.add_argument("--exp1", action="store_true",
                        help="Run §3.2 — PMC supervised tables (Tables 2, 3).")
    parser.add_argument("--exp2", action="store_true",
                        help="Run §3.3 — PMM tables (Tables 4, 5).")
    parser.add_argument("--exp3", action="store_true",
                        help="Run §4.3 — ICE selection (Tables 6, 7).")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true",
                   help="Reduced settings: 30 reps for exp1/2, 5 ICE runs (default).")
    g.add_argument("--full", action="store_true",
                   help="Paper settings: 300 reps and 10 ICE runs (~3h total).")
    parser.add_argument("--reps",
                        type=int, default=None,
                        help="Override the number of repetitions for exp1/2.")
    parser.add_argument("--runs",
                        type=int, default=None,
                        help="Override the number of ICE runs for exp3.")
    parser.add_argument("--n-obs", type=int, default=2000,
                        help="Sequence length for exp1/2 (default 2000).")
    parser.add_argument("--jobs", type=int, default=None,
                        help="Number of worker processes for parallel reps "
                             "and ICE runs. Default: half the available "
                             "CPUs (currently %d on this machine). "
                             "Pass 1 to disable parallelism."
                             % _default_jobs())
    args = parser.parse_args(argv)

    # Defaults: --quick if neither --quick nor --full given.
    if args.full:
        reps_default = 300
        runs_default = 10
    else:
        reps_default = 30
        runs_default = 5
    reps = args.reps if args.reps is not None else reps_default
    runs = args.runs if args.runs is not None else runs_default
    jobs = args.jobs if args.jobs is not None else _default_jobs()

    do_all = not (args.exp1 or args.exp2 or args.exp3)

    # Parallel pool — created once and shared across the three experiments
    # so we pay the (~few-hundred-ms) worker spin-up cost only once.
    # ``jobs == 1`` disables parallelism entirely (sequential path); useful
    # for debugging and for users who want byte-for-byte reproducibility
    # of the in-order log output.
    executor: ProcessPoolExecutor | None = None
    if jobs > 1:
        logger.info("Parallel mode: %d worker processes", jobs)
        executor = ProcessPoolExecutor(max_workers=jobs)
    else:
        logger.info("Sequential mode (--jobs 1)")

    try:
        res1 = res2 = res3 = None
        if do_all or args.exp1:
            res1 = run_exp1(reps=reps, n_obs=args.n_obs, executor=executor)
        if do_all or args.exp2:
            res2 = run_exp2(reps=reps, n_obs=args.n_obs, executor=executor)
        if do_all or args.exp3:
            res3 = run_exp3(runs=runs, executor=executor)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    write_all_latex_tables(res1, res2, res3)

    # Auto-generated metadata so the report's prose tracks the actual
    # run parameters (reps / runs / mode), regardless of --quick vs --full.
    if args.full:
        mode = "full"
    elif args.quick:
        mode = "quick"
    elif args.reps is not None or args.runs is not None or args.n_obs != 2000:
        mode = "custom"
    else:
        mode = "quick"
    _save_run_metadata(
        reps=reps, runs=runs, n_obs=args.n_obs, mode=mode,
        exp1=do_all or args.exp1,
        exp2=do_all or args.exp2,
        exp3=do_all or args.exp3,
    )
    logger.info("Done.  CSV → %s   LaTeX → %s", RESULTS_DIR, TABLES_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
