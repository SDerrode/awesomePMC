#!/usr/bin/env python3
"""Is the multivariate KS test usable as a post-fit specification test?
(research opportunity O7)

The question
------------
``pmcprg.diagnostics.mks`` compares a sample against a *known* distribution and
takes its critical value from Naaman's finite-sample bound — a multivariate
DKW inequality, valid for an i.i.d. sample. Applied to a **fitted** PMC model
three of its assumptions break at once, in opposite directions:

* **Durbin effect** — the reference distribution is estimated from the same
  data, which makes the statistic stochastically *smaller*, i.e. pushes the
  test conservative;
* **serial dependence** — Y is a Markov chain, so its per-state slices are
  not i.i.d., which inflates the statistic and pushes the test
  *anti*-conservative;
* **label uncertainty**, specific to a latent-state model — the per-state
  sample has to be obtained somehow, and how one obtains it changes the null.

Which effect wins is an empirical question, and the answer was unknown. A
first measurement made while wiring the GUI (120 datasets, N=600) found the
tabulated test rejecting a *correct* model 9.2% of the time at a nominal 5%
— serial dependence dominating — and found MPM assignment biasing the
statistic upward (0.0828 / 0.0895) where a posterior draw tracked the
true-label reference (0.0790 / 0.0781 vs 0.0785 / 0.0813). This script turns
those two observations into a proper study.

The design
----------
Six variants, from the crossing of

* **critical value**: ``tabulated`` (Naaman) vs ``bootstrap`` (parametric,
  simulating whole series from the fitted model and running the identical
  pipeline — the same device that calibrated the GoF test);
* **labels**: ``oracle`` (the true X, available only in simulation and used
  as the reference), ``draw`` (one FFBS draw from P(X | Y)), ``mpm``.

against five data-generating configurations, all tested against the same
normal-margin reference model:

* ``h0``      — the reference itself: this is the **level**;
* ``shift``   — state 0's location moved by +0.5;
* ``scale``   — state 0's dispersion multiplied by 1.35;
* ``student`` — Student-t margins, **matched in mean and variance**, so only
  the tails differ. The analogue of the matched-τ design in the GoF study,
  and the honest hard case;
* ``skew``    — skew-normal margins, matched in mean and variance.

Matching the first two moments is what makes ``student`` and ``skew``
informative: a test that only notices a moved mean is not a specification
test, it is a t-test.

Cost and interruption
---------------------
~130 ms per bootstrap replicate at N = 600, so ~26 s per dataset at B = 200.
Tasks are emitted replicate-major so the study stays balanced at any moment,
each result is flushed as it lands, and re-running skips what is recorded.

Usage
-----
    python report/mks_specification_study.py --reps 200 --B 200 --workers 8
    python report/mks_specification_study.py --report-only
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = REPORT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_DIR  = REPORT_DIR / "out" / "mks_specification"
OUT_FILE = OUT_DIR / "results.jsonl"

BASE_MODEL = REPO_ROOT / "pmcprg" / "pmc" / "models" / "pmc_gauss_k2.toml"

CONFIGS = ["h0", "shift", "scale", "student", "skew"]
LABELS  = ["oracle", "draw", "mpm"]
ALPHA   = 0.05
N_DEFAULT = 600


# ---------------------------------------------------------------------------
# Generating models — alternatives matched in mean and variance where it counts
# ---------------------------------------------------------------------------

def _student_params(loc: float, scale: float, df: float = 4.0) -> dict:
    """Student-t matched to ``N(loc, scale²)`` in mean and variance.

    Var(t_df) = df/(df−2), so the scale must be divided by its square root.
    Without this the alternative would differ in spread as well as in tails
    and the study would not be measuring what it claims to.
    """
    return {"df": df, "loc": loc, "scale": scale / math.sqrt(df / (df - 2.0))}


def _skewnorm_params(loc: float, scale: float, a: float = 4.0) -> dict:
    """Skew-normal matched to ``N(loc, scale²)`` in mean and variance."""
    delta = a / math.sqrt(1.0 + a * a)
    m     = delta * math.sqrt(2.0 / math.pi)
    v     = 1.0 - 2.0 * delta * delta / math.pi
    s     = scale / math.sqrt(v)
    return {"a": a, "loc": loc - m * s, "scale": s}


def build_models(config: str):
    """(generating model, reference model under test)."""
    from pmcprg.pmc.model import PMCModel

    reference = PMCModel(BASE_MODEL)
    raw = reference.raw
    # "State 0" is every margin attached to x_n = 0: the block f_0 for state
    # margins, the blocks f_0j (j = 0..K−1) for the pair margins of a general
    # PMC (A16 Eqs. 12–14). The distributional alternatives change them all.
    blocks = raw["margins"]
    state0 = [blk for blk in blocks if int(blk["i"]) == 0]

    if config == "h0":
        pass
    elif config == "shift":
        for blk in state0:
            blk["params"]["loc"] += 0.5
    elif config == "scale":
        for blk in state0:
            blk["params"]["scale"] *= 1.35
    elif config == "student":
        for blk in blocks:
            p = blk["params"]
            blk["dist"], blk["params"] = "t", _student_params(p["loc"], p["scale"])
    elif config == "skew":
        for blk in blocks:
            p = blk["params"]
            blk["dist"] = "skewnorm"
            blk["params"] = _skewnorm_params(p["loc"], p["scale"])
    else:
        raise ValueError(config)
    return PMCModel.from_dict(raw), reference


# ---------------------------------------------------------------------------
# One dataset
# ---------------------------------------------------------------------------

def _labelled_stats(mdl, Yv, X_true, rng):
    """MKS statistic per (label scheme, margin), plus the tabulated verdict.

    A margin is a state k (state margins) or a pair (i, j) (pair margins of a
    general PMC, A16 Eqs. 12–14, sample in the dual view — see
    :func:`pmcprg.diagnostics.margin_sample`).

    All three schemes come out of one forward-backward pass: the replicate
    cost is dominated by that pass, so measuring the three together costs
    barely more than measuring one, and they then share the same data.
    """
    import numpy as np

    from pmcprg.diagnostics   import (
        margin_keys, margin_of, margin_sample, mks_1samp,
    )
    from pmcprg.pmc.inference import (
        backward, forward, precompute_weights, smooth,
    )
    from pmcprg.pmc.inference import mpm as _mpm
    from pmcprg.pmc.inference import sample_posterior

    W, f_pdf = precompute_weights(mdl, Yv)
    alpha, _ = forward(mdl, Yv, W=W, f_pdf=f_pdf)
    beta     = backward(mdl, Yv, W=W)
    gamma    = smooth(alpha, beta)
    schemes = {
        "oracle": np.asarray(X_true),
        "draw":   sample_posterior(mdl, Yv, rng, W=W, f_pdf=f_pdf,
                                   alpha_hat=alpha),
        "mpm":    _mpm(gamma),
    }
    out = {}
    for name, lab in schemes.items():
        for k in margin_keys(mdl):
            sel = margin_sample(Yv, lab, k)
            if len(sel) < 20:
                out[(name, k)] = (float("nan"), float("nan"), len(sel))
                continue
            arr = sel.reshape(-1, 1) if sel.ndim == 1 else sel
            try:
                def _cdf(point, _k=k):
                    return float(margin_of(mdl, _k).cdf_vec(
                        np.atleast_1d(np.asarray(point, dtype=float)))[0])
                res = mks_1samp(arr, cdf=_cdf, alpha=ALPHA)
                out[(name, k)] = (float(res.statistic),
                                  float(res.critical_value), len(sel))
            except Exception:
                out[(name, k)] = (float("nan"), float("nan"), len(sel))
    return out


def run_one(task) -> dict:
    """Simulate under `config`, test against the normal-margin reference."""
    import numpy as np

    from pmcprg.pmc.simulate import simulate

    config, N, rep, B = task
    gen, ref = build_models(config)

    t0 = time.perf_counter()
    X, Y = simulate(gen, N=N, seed=500_000 + 1000 * rep + N)
    rng  = np.random.default_rng(rep)
    observed = _labelled_stats(ref, Y, X, rng)

    # Parametric bootstrap: whole series from the REFERENCE model, identical
    # pipeline. The replicate's own true labels feed the `oracle` scheme, so
    # one loop calibrates all three label schemes at once.
    null: dict = {key: [] for key in observed}
    for _ in range(B):
        try:
            Xb, Yb = simulate(ref, N=N,
                              seed=int(rng.integers(0, 2**31 - 1)))
            for key, (s, _cv, _m) in _labelled_stats(ref, Yb, Xb, rng).items():
                if np.isfinite(s):
                    null[key].append(s)
        except Exception:
            pass

    rows = []
    for (scheme, k), (stat, crit, m) in observed.items():
        draws = np.asarray(null[(scheme, k)], dtype=float)
        rows.append({
            "scheme": scheme,
            **({"pair": list(k)} if isinstance(k, tuple) else {"state": k}),
            "n": int(m),
            "statistic": (float(stat) if np.isfinite(stat) else None),
            "critical": (float(crit) if np.isfinite(crit) else None),
            "reject_tabulated": (bool(stat > crit)
                                 if np.isfinite(stat) and np.isfinite(crit)
                                 else None),
            "p_bootstrap": (float(np.mean(draws >= stat))
                            if draws.size and np.isfinite(stat) else None),
            "n_valid": int(draws.size),
        })
    return {"config": config, "N": N, "rep": rep, "B": B,
            "seconds": round(time.perf_counter() - t0, 2), "rows": rows}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96):
    """Wilson (1927) score interval for a binomial proportion.

    Reference: Wilson, E. B. (1927). Probable inference, the law of
    succession, and statistical inference. *JASA* 22(158), 209–212.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def summarise(rows_all) -> None:
    import numpy as np

    acc: dict = {}
    for row in rows_all:
        for r in row["rows"]:
            key = (row["config"], r["scheme"])
            slot = acc.setdefault(key, {"tab": [], "boot": [], "stat": []})
            if r["reject_tabulated"] is not None:
                slot["tab"].append(bool(r["reject_tabulated"]))
            if r["p_bootstrap"] is not None:
                slot["boot"].append(float(r["p_bootstrap"]) < ALPHA)
            if r["statistic"] is not None:
                slot["stat"].append(float(r["statistic"]))

    print(f"\nTaux de rejet au seuil {ALPHA:.0%} — `h0` mesure le NIVEAU, "
          f"les autres la PUISSANCE")
    print(f"{'config':<10}{'étiquettes':<10}{'n':>6}"
          f"{'tabulé':>10}{'Wilson':>18}{'bootstrap':>11}{'Wilson':>18}"
          f"{'stat méd':>10}")
    print("-" * 93)
    for config in CONFIGS:
        for scheme in LABELS:
            slot = acc.get((config, scheme))
            if not slot or not slot["tab"]:
                continue
            nt = len(slot["tab"]); kt = int(np.sum(slot["tab"]))
            nb = len(slot["boot"]); kb = int(np.sum(slot["boot"]))
            lo_t, hi_t = wilson(kt, nt)
            lo_b, hi_b = wilson(kb, nb) if nb else (float("nan"),) * 2
            print(f"{config:<10}{scheme:<10}{nt:>6}"
                  f"{kt / nt:>10.3f}{f'[{lo_t:.3f}, {hi_t:.3f}]':>18}"
                  f"{(kb / nb if nb else float('nan')):>11.3f}"
                  f"{f'[{lo_b:.3f}, {hi_b:.3f}]':>18}"
                  f"{np.median(slot['stat']):>10.4f}")
        print()


def load(path: Path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--N", type=int, default=N_DEFAULT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load(OUT_FILE)
    if args.report_only:
        print(f"{len(rows)} datasets in {OUT_FILE}")
        summarise(rows)
        return

    done = {(r["config"], r["N"], r["rep"], r.get("B")) for r in rows}
    wanted = [c for c in args.configs.split(",") if c in CONFIGS]
    tasks = [
        (config, args.N, rep, args.B)
        for rep in range(args.reps)
        for config in wanted
        if (config, args.N, rep, args.B) not in done
    ]
    if not tasks:
        print("Grid complete.")
        summarise([r for r in rows if r.get("B") == args.B])
        return

    print(f"{len(tasks)} datasets, B={args.B}, N={args.N}, "
          f"{args.workers} workers -> {OUT_FILE}")
    t0 = time.perf_counter()
    with OUT_FILE.open("a") as fh, \
            ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, t): t for t in tasks}
        for n_done, fut in enumerate(as_completed(futures), 1):
            try:
                row = fut.result()
            except Exception as exc:
                print(f"  !! {futures[fut][:3]} : {type(exc).__name__}: {exc}",
                      flush=True)
                continue
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if n_done % 20 == 0 or n_done == len(tasks):
                el = time.perf_counter() - t0
                print(f"  {n_done}/{len(tasks)}  {el / 60:.1f} min  "
                      f"eta {el / n_done * (len(tasks) - n_done) / 60:.1f} min",
                      flush=True)
    summarise([r for r in load(OUT_FILE) if r.get("B") == args.B])


if __name__ == "__main__":           # required under the spawn start method
    main()
