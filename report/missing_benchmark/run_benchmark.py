"""
run_benchmark.py — classification and imputation under missing data, known models.

For every model × missingness pattern × missing rate × replicate, a sequence
(X, Y) is simulated from a *known* model, Y is masked with a
:mod:`pmcprg.missing.patterns` generator, and six methods classify the states
(and, where they can, impute the missing values) with the **true** model:

``complete``
    ``classify`` on the full Y — the lower bound on the error, no imputation.
``marginalise``
    ``classify(model, Y_masked)``: the missing values are integrated out
    (exact shortcut for HMC-IN, quadrature grid otherwise). Imputation from
    ``impute``: posterior mean and sd, quantiles 2.5/5/50/95/97.5 %, 200
    joint FFBS draws for the CRPS.
``plugin``
    Posterior mean of ``impute`` written into Y as if observed, then
    ``classify``. Imputation: the same mean with a Gaussian law N(mean, sd²)
    of the exact posterior sd (Gaussian CRPS, ±1.645 sd and ±1.96 sd bands).
``linear``
    Linear interpolation between the neighbouring observed values (nearest
    observed value before the first / after the last one), then ``classify``.
``locf``
    Last observation carried forward (the first observed value carried
    backward over a leading gap), then ``classify``.
``mean``
    Mean of the observed values, then ``classify``.

Scores (:mod:`pmcprg.missing.metrics`)
--------------------------------------
Classification — ``error_rate_split`` (missing / observed / all positions,
Hungarian label alignment); on the missing positions also the mean posterior
confidence max_k γ_n(k), the Brier score Σ_k (γ_n(k) − 1{x_n = k})² and the
log-loss −log γ_n(x_n) (γ clipped at 1e-12), on the true labels.
``loglik`` is what ``classify`` returns: log p(y_obs) for ``marginalise``
(``loglik_kind = observed-data``), log p(y) for ``complete`` and the
log-likelihood of the *filled* series for the fill-in methods
(``filled-series`` — not a likelihood of the data, not comparable).

Imputation, on the missing positions — RMSE and MAE of the point imputation;
CRPS (``crps_kind``): ``samples`` (``crps_from_samples`` on the FFBS draws —
drawn on the quadrature nodes for the grid variants) for ``marginalise``,
``gaussian`` (``crps_gaussian`` with the exact posterior sd) for ``plugin``,
``gaussian-sd-obs`` for the point methods (N(point, s²) with s the standard
deviation of the observed values; the CRPS of the point itself, a Dirac, is
the MAE). Coverage and mean width of the 90 % and 95 % intervals: posterior
quantiles [q05, q95] and [q025, q975] for ``marginalise``; point ± 1.645 sd
and ± 1.96 sd for ``plugin`` (posterior sd) and the point methods (s).
Wall time per method: filling / imputation, classification, total.

Seeds
-----
Keyed by names, not by loop positions: the simulation depends on (base seed,
model, rep) only — every pattern and rate of a replicate masks the *same*
sequence (common random numbers) — and the mask and the FFBS draws on (base
seed, model, pattern, rate, rep). Adding a model or a rate changes no other
cell; ``--jobs`` does not change any number.

Usage (from the repository root)
--------------------------------
::

    # Smoke run: 3 reps, rates 0.1 and 0.4, N = 1000 — about 15 s with
    # 4 processes; writes to report/results/missing_benchmark/quick/.
    .venv/bin/python report/missing_benchmark/run_benchmark.py --quick --jobs 4

    # Full benchmark (defaults: 5 models, 5 patterns, 5 rates, 20 reps,
    # N = 2000): about 6 min with 4 processes.
    .venv/bin/python report/missing_benchmark/run_benchmark.py --jobs 4
    .venv/bin/python report/missing_benchmark/summarise.py

Outputs (``--out``, default ``report/results/missing_benchmark/``)
-----------------------------------------------------------------
``runs.csv``      one row per model / pattern / rate / rep / method.
``summary.csv``   mean and sd over the reps of every score.
``run_info.json`` settings, versions, commit, wall time.
"""

from __future__ import annotations

import os

# One BLAS thread per worker: the processes are the parallelism.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402
from collections.abc import Iterable  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPORT_DIR = HERE.parent
ROOT = REPORT_DIR.parent
for _p in (ROOT, REPORT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import scipy  # noqa: E402
from scipy.stats import norm  # noqa: E402

from pmcprg.missing import metrics, patterns  # noqa: E402
from pmcprg.pmc import PMCModel, classify, impute, simulate  # noqa: E402
from pmcprg.pmc.gaps import needs_grid  # noqa: E402
from reproduce_csda2013 import MARGIN_SETS, _progress  # noqa: E402

logger = logging.getLogger("missing_benchmark")
logging.getLogger("pmcprg").setLevel(logging.WARNING)

RESULTS_DIR = REPORT_DIR / "results" / "missing_benchmark"
MODELS_DIR = ROOT / "pmcprg" / "pmc" / "models"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

#: key → (label, TOML file or None for a model built in code).
MODELS: dict[str, tuple[str, str | None]] = {
    "hmc_in_gauss_k2": ("HMC-IN, Gaussian (exact shortcut)", "hmc_in_gauss_k2.toml"),
    "hmc_dn_gauss_k2": ("HMC-DN, Gaussian copulas", "hmc_dn_gauss_k2.toml"),
    "pmc_gauss_k2": ("PMC, state margins, Gaussian copulas", "pmc_gauss_k2.toml"),
    "pmc_pair_gauss_k2": ("PMC, pair Gaussian margins, Clayton", "pmc_pair_gauss_k2.toml"),
    "pmc_pair_gamma_gumbel_k2": ("PMC, pair Gamma margins, Gumbel", None),
}

#: Joint prior and Kendall's τ of the Gamma–Gumbel pair model (CSDA 2013 Table 1,
#: as in pmc_pair_gauss_k2.toml).
GAMMA_GUMBEL_PRIOR = [[0.50, 0.05], [0.05, 0.40]]
GAMMA_GUMBEL_TAU = 0.7


def _gamma_gumbel_model() -> PMCModel:
    """General PMC: Table 1 pair Gamma margins (``MARGIN_SETS["pair"]["gamma"]``,
    s read as a variance) and Gumbel–Hougaard copulas τ = 0.7 on the four pairs."""
    raw = {
        "model": {"name": "PMC pair Gamma-Gumbel K=2", "variant": "PMC", "K": 2,
                  "N_default": 2000},
        "prior": {"p": GAMMA_GUMBEL_PRIOR},
        "margins": [dict(b) for b in MARGIN_SETS["pair"]["gamma"]],
        "copulas": [{"i": i, "j": j, "name": "GH", "tau": GAMMA_GUMBEL_TAU}
                    for i in range(2) for j in range(2)],
    }
    return PMCModel.from_dict(raw)


_MODEL_CACHE: dict[str, PMCModel] = {}


def get_model(key: str) -> PMCModel:
    """The model ``key``, built once per process."""
    if key not in _MODEL_CACHE:
        if key not in MODELS:
            raise ValueError(f"unknown model {key!r}; choose from {tuple(MODELS)}")
        toml = MODELS[key][1]
        _MODEL_CACHE[key] = (_gamma_gumbel_model() if toml is None
                             else PMCModel(str(MODELS_DIR / toml)))
    return _MODEL_CACHE[key]


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

#: key → label. Every pattern keeps GenGap's protected offset (first 10 %).
PATTERNS: dict[str, str] = {
    "mcar_b1": "MCAR, blocks of 1",
    "mcar_b10": "MCAR, blocks of 10",
    "mcar_b50": "MCAR, blocks of 50",
    "scattered": "scattered (one block, random start)",
    "aligned": "aligned (one block after the offset)",
}


def apply_pattern(key: str, Y: np.ndarray, rate: float, seed: int):
    """(Y_masked, mask) of pattern ``key`` at missing rate ``rate``."""
    if key.startswith("mcar_b"):
        return patterns.mcar(Y, rate, block_size=int(key[len("mcar_b"):]), seed=seed)
    if key == "scattered":
        return patterns.scattered(Y, rate, seed=seed)
    if key == "aligned":
        return patterns.aligned(Y, rate)
    raise ValueError(f"unknown pattern {key!r}; choose from {tuple(PATTERNS)}")


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

METHODS: tuple[str, ...] = ("complete", "marginalise", "plugin", "linear", "locf", "mean")

QUANTILES = (0.025, 0.05, 0.5, 0.95, 0.975)
Z90 = float(norm.ppf(0.95))    # 1.645
Z95 = float(norm.ppf(0.975))   # 1.960
GAMMA_FLOOR = 1e-12


def fill_linear(Y_gap: np.ndarray, miss: np.ndarray) -> np.ndarray:
    idx = np.arange(Y_gap.size)
    out = Y_gap.copy()
    out[miss] = np.interp(idx[miss], idx[~miss], Y_gap[~miss])   # nearest at the edges
    return out


def fill_locf(Y_gap: np.ndarray, miss: np.ndarray) -> np.ndarray:
    idx = np.arange(Y_gap.size)
    last = np.maximum.accumulate(np.where(miss, -1, idx))
    last[last < 0] = int(np.flatnonzero(~miss)[0])                 # leading gap: first value
    return Y_gap[last]


def fill_mean(Y_gap: np.ndarray, miss: np.ndarray) -> np.ndarray:
    out = Y_gap.copy()
    out[miss] = float(Y_gap[~miss].mean())
    return out


FILLERS = {"linear": fill_linear, "locf": fill_locf, "mean": fill_mean}


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

ID_COLUMNS = ("model", "pattern", "rate", "rep", "method")
INFO_COLUMNS = ("n", "n_missing", "n_gaps", "route", "loglik_kind", "crps_kind",
                "seed_sim", "seed_mask")
METRIC_COLUMNS = (
    "err_missing", "err_observed", "err_overall",
    "conf_missing", "brier_missing", "logloss_missing", "loglik",
    "rmse", "mae", "crps", "cov90", "width90", "cov95", "width95",
    "time_fill_s", "time_classify_s", "time_total_s",
)
COLUMNS = ID_COLUMNS + INFO_COLUMNS + METRIC_COLUMNS


def _classification_scores(X, X_hat, gamma, mask) -> dict:
    er = metrics.error_rate_split(X, X_hat, mask)
    g = np.asarray(gamma, dtype=float)[mask]
    x = np.asarray(X)[mask]
    p_true = g[np.arange(x.size), x]
    onehot = np.zeros_like(g)
    onehot[np.arange(x.size), x] = 1.0
    return {
        "err_missing": er.missing, "err_observed": er.observed, "err_overall": er.overall,
        "conf_missing": float(g.max(axis=1).mean()),
        "brier_missing": float(((g - onehot) ** 2).sum(axis=1).mean()),
        "logloss_missing": float(-np.log(np.maximum(p_true, GAMMA_FLOOR)).mean()),
    }


def _band_scores(Y, centre, half90, half95, mask) -> dict:
    """Coverage and mean width of centre ± half-widths (full-length arrays)."""
    return {
        "cov90": metrics.interval_coverage(Y, centre - half90, centre + half90, mask=mask),
        "width90": float(2.0 * np.mean(np.broadcast_to(half90, Y.shape)[mask])),
        "cov95": metrics.interval_coverage(Y, centre - half95, centre + half95, mask=mask),
        "width95": float(2.0 * np.mean(np.broadcast_to(half95, Y.shape)[mask])),
    }


def _seed(*parts) -> int:
    """32-bit seed from a tuple of ints and strings (names hashed with CRC-32)."""
    words = [zlib.crc32(p.encode()) if isinstance(p, str) else int(p) for p in parts]
    return int(np.random.SeedSequence(words).generate_state(1, dtype=np.uint32)[0])


def _rate_key(rate: float) -> int:
    return int(round(rate * 10000))


@dataclass(frozen=True)
class Task:
    model: str
    pattern: str
    rate: float
    rep: int
    n_obs: int
    base_seed: int
    n_draws: int


def run_task(task: Task) -> list[dict]:
    """All methods on one simulated, masked sequence — pickle-safe worker."""
    mdl = get_model(task.model)
    seed_sim = _seed("simulate", task.base_seed, task.model, task.rep)
    seed_mask = _seed("mask", task.base_seed, task.model, task.pattern,
                      _rate_key(task.rate), task.rep)
    seed_draw = _seed("draws", task.base_seed, task.model, task.pattern,
                      _rate_key(task.rate), task.rep)
    X, Y = simulate(mdl, N=task.n_obs, seed=seed_sim)
    Y = np.asarray(Y, dtype=float)
    Y_gap, mask = apply_pattern(task.pattern, Y, task.rate, seed_mask)
    mask = np.asarray(mask, dtype=bool)
    N = Y.size
    s_obs = float(np.std(Y_gap[~mask], ddof=1))
    base = {
        "model": task.model, "pattern": task.pattern, "rate": task.rate, "rep": task.rep,
        "n": N, "n_missing": int(mask.sum()),
        "n_gaps": int(np.count_nonzero(np.diff(mask.astype(int), prepend=0) == 1)),
        "route": "grid" if needs_grid(mdl) else "exact",
        "seed_sim": seed_sim, "seed_mask": seed_mask,
    }
    rows: dict[str, dict] = {}

    def record(method, X_hat, gamma, loglik, loglik_kind, t_fill, t_cls, extra=None):
        row = {c: math.nan for c in METRIC_COLUMNS}
        row.update(base, method=method, loglik=float(loglik), loglik_kind=loglik_kind,
                   crps_kind="", time_fill_s=t_fill, time_classify_s=t_cls,
                   time_total_s=t_fill + t_cls)
        row.update(_classification_scores(X, X_hat, gamma, mask))
        if extra:
            row.update(extra)
        rows[method] = row

    def timed(fn, *args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        return out, time.perf_counter() - t0

    # complete — lower bound
    (X_hat, gamma, ll), t_cls = timed(classify, mdl, Y)
    record("complete", X_hat, gamma, ll, "complete-data", 0.0, t_cls)

    # marginalise — exact integration of the missing values
    (X_hat, gamma, ll), t_cls = timed(classify, mdl, Y_gap)
    imp, t_imp = timed(impute, mdl, Y_gap, quantiles=QUANTILES, n_samples=task.n_draws,
                       rng=seed_draw)
    idx = np.flatnonzero(mask)
    if not np.array_equal(imp.index, idx):
        raise RuntimeError("impute().index differs from the pattern mask")
    mean_full = np.zeros(N)
    mean_full[idx] = imp.mean
    q = {lvl: np.zeros(N) for lvl in QUANTILES}
    for k, lvl in enumerate(QUANTILES):
        q[lvl][idx] = imp.quantile_values[:, k]
    draws = np.zeros((N, task.n_draws))
    draws[idx] = np.asarray(imp.y_samples).T
    extra = {
        "rmse": metrics.rmse(Y, mean_full, mask), "mae": metrics.mae(Y, mean_full, mask),
        "crps": metrics.crps_from_samples(Y, draws, mask=mask), "crps_kind": "samples",
        "cov90": metrics.interval_coverage(Y, q[0.05], q[0.95], mask=mask),
        "width90": float(np.mean(q[0.95][idx] - q[0.05][idx])),
        "cov95": metrics.interval_coverage(Y, q[0.025], q[0.975], mask=mask),
        "width95": float(np.mean(q[0.975][idx] - q[0.025][idx])),
    }
    # impute() was called with quantiles and draws: its time is the whole
    # probabilistic imputation, classification is timed separately.
    record("marginalise", X_hat, gamma, ll, "observed-data", t_imp, t_cls, extra)

    # plugin — posterior mean treated as observed
    imp0, t_imp0 = timed(impute, mdl, Y_gap, quantiles=(), n_samples=0)
    sd_full = np.ones(N)
    sd_full[idx] = imp0.sd
    centre = np.array(imp0.Y_mean, dtype=float)
    (X_hat, gamma, ll), t_cls = timed(classify, mdl, centre)
    extra = {
        "rmse": metrics.rmse(Y, centre, mask), "mae": metrics.mae(Y, centre, mask),
        "crps": metrics.crps_gaussian(Y, centre, sd_full, mask=mask), "crps_kind": "gaussian",
        **_band_scores(Y, centre, Z90 * sd_full, Z95 * sd_full, mask),
    }
    record("plugin", X_hat, gamma, ll, "filled-series", t_imp0, t_cls, extra)

    # fill-in baselines
    for method, filler in FILLERS.items():
        filled, t_fill = timed(filler, Y_gap, mask)
        (X_hat, gamma, ll), t_cls = timed(classify, mdl, filled)
        extra = {
            "rmse": metrics.rmse(Y, filled, mask), "mae": metrics.mae(Y, filled, mask),
            "crps": metrics.crps_gaussian(Y, filled, s_obs, mask=mask),
            "crps_kind": "gaussian-sd-obs",
            **_band_scores(Y, filled, Z90 * s_obs, Z95 * s_obs, mask),
        }
        record(method, X_hat, gamma, ll, "filled-series", t_fill, t_cls, extra)

    return [rows[m] for m in METHODS]


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def _fmt(value) -> str:
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.6g}"
    return str(value)


def read_runs(path: Path) -> list[dict]:
    """Rows of ``runs.csv`` with numbers parsed."""
    out = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            row["rate"] = float(row["rate"])
            row["rep"] = int(row["rep"])
            for c in METRIC_COLUMNS:
                row[c] = float(row[c])
            out.append(row)
    return out


def summarise_runs(rows: Iterable[dict]) -> list[dict]:
    """Mean and sd (ddof = 1) over the reps, per model / pattern / rate / method."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["model"], r["pattern"], float(r["rate"]), r["method"]), []).append(r)
    out = []
    for (model, pattern, rate, method), rs in groups.items():
        s = {"model": model, "pattern": pattern, "rate": rate, "method": method,
             "n_reps": len(rs), "loglik_kind": rs[0]["loglik_kind"],
             "crps_kind": rs[0]["crps_kind"]}
        for c in ("n_missing", "n_gaps"):
            s[f"{c}_mean"] = float(np.mean([float(r[c]) for r in rs]))
        for c in METRIC_COLUMNS:
            v = np.array([float(r[c]) for r in rs])
            v = v[np.isfinite(v)]
            s[f"{c}_mean"] = float(v.mean()) if v.size else math.nan
            s[f"{c}_sd"] = float(v.std(ddof=1)) if v.size > 1 else math.nan
        out.append(s)
    return out


def write_summary(summary: list[dict], path: Path) -> None:
    cols = list(summary[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for s in summary:
            w.writerow([_fmt(s[c]) for c in cols])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_RATES = (0.05, 0.1, 0.2, 0.4, 0.6)
QUICK = {"reps": 3, "rates": (0.1, 0.4), "n_obs": 1000}


def _csv_list(text: str, allowed: Iterable[str], what: str) -> list[str]:
    allowed = tuple(allowed)
    items = [t.strip() for t in text.split(",") if t.strip()]
    bad = [t for t in items if t not in allowed]
    if bad or not items:
        raise argparse.ArgumentTypeError(f"{what} must be among {allowed}, got {text!r}")
    return items


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s | %(message)s", force=True)
    logging.getLogger("pmcprg").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(MODELS),
                    type=lambda s: _csv_list(s, MODELS, "--models"),
                    help="comma-separated model keys (default: all five)")
    ap.add_argument("--patterns", default=",".join(PATTERNS),
                    type=lambda s: _csv_list(s, PATTERNS, "--patterns"),
                    help="comma-separated pattern keys (default: all five)")
    ap.add_argument("--rates", default=None,
                    help="comma-separated missing rates in (0, 1) "
                         f"(default {','.join(map(str, DEFAULT_RATES))})")
    ap.add_argument("--reps", type=int, default=None, help="replicates per cell (default 20)")
    ap.add_argument("--n-obs", type=int, default=None, help="sequence length (default 2000)")
    ap.add_argument("--draws", type=int, default=200,
                    help="FFBS draws of the missing values for the CRPS (default 200)")
    ap.add_argument("--seed", type=int, default=20260915, help="base seed")
    ap.add_argument("--jobs", type=int, default=None,
                    help="worker processes (default min(4, half the CPUs)); 1 = sequential")
    ap.add_argument("--quick", action="store_true",
                    help=f"smoke run: {QUICK['reps']} reps, rates "
                         f"{','.join(map(str, QUICK['rates']))}, N = {QUICK['n_obs']}; "
                         "writes to <results>/quick unless --out is given")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"output directory (default {RESULTS_DIR.relative_to(ROOT)})")
    args = ap.parse_args(argv)

    reps = args.reps if args.reps is not None else (QUICK["reps"] if args.quick else 20)
    n_obs = args.n_obs if args.n_obs is not None else (QUICK["n_obs"] if args.quick else 2000)
    rates = (tuple(float(r) for r in args.rates.split(",")) if args.rates
             else (QUICK["rates"] if args.quick else DEFAULT_RATES))
    if any(not 0.0 < r < 1.0 for r in rates):
        ap.error(f"rates must lie in (0, 1), got {rates}")
    jobs = args.jobs if args.jobs is not None else max(1, min(4, (os.cpu_count() or 2) // 2))
    out = args.out if args.out is not None else (RESULTS_DIR / "quick" if args.quick
                                                 else RESULTS_DIR)
    out.mkdir(parents=True, exist_ok=True)

    tasks = [Task(m, p, r, rep, n_obs, args.seed, args.draws)
             for m in args.models for p in args.patterns for r in rates for rep in range(reps)]
    logger.info("%d models × %d patterns × %d rates × %d reps = %d tasks, N = %d, %d job(s)",
                len(args.models), len(args.patterns), len(rates), reps, len(tasks), n_obs, jobs)

    runs_path = out / "runs.csv"
    rows: list[dict] = []
    t0 = time.perf_counter()
    executor = ProcessPoolExecutor(max_workers=jobs) if jobs > 1 else None
    try:
        source = (executor.map(run_task, tasks, chunksize=2) if executor is not None
                  else map(run_task, tasks))
        with open(runs_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(COLUMNS)
            for task_rows in _progress(source, total=len(tasks), desc="  missing benchmark"):
                for r in task_rows:
                    writer.writerow([_fmt(r[c]) for c in COLUMNS])
                rows.extend(task_rows)
                fh.flush()
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    wall = time.perf_counter() - t0

    write_summary(summarise_runs(rows), out / "summary.csv")
    info = {
        "models": args.models, "patterns": args.patterns, "rates": list(rates), "reps": reps,
        "n_obs": n_obs, "draws": args.draws, "quantiles": list(QUANTILES), "seed": args.seed,
        "jobs": jobs, "quick": bool(args.quick), "n_tasks": len(tasks),
        "wall_time_s": round(wall, 1),
        "cpu_time_s": round(sum(r["time_total_s"] for r in rows), 1),
        "commit": _git_commit(),
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "platform": platform.platform(),
    }
    (out / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    logger.info("Done in %.0f s → %s", wall, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
