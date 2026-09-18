#!/usr/bin/env python3
"""Which statistic should a margin specification test use? (opportunity O7)

``report/mks_specification_study.py`` asks whether the multivariate KS test
can be calibrated post-fit. This script asks the next question, and the one
that S-6 taught to ask first: *is KS the right statistic at all?*

The lesson from the copula side is blunt. There, reweighting the
Cramér-von Mises statistic — the thing that looked like the problem — bought
nothing, while swapping the statistic for one on the Kendall process took
rejection of a wrong family from 0.115 to 0.950. Calibration and power are
different questions, and the second one is answered by the choice of
statistic.

The design
----------
**Every statistic is calibrated by the same parametric bootstrap**, so every
one of them has the correct level by construction and the comparison is
purely about power. That is the whole point of the panel: comparing
*tabulated* tests would confound a statistic's sensitivity with the accuracy
of whatever table it ships with.

Labels are fixed to a posterior draw — the scheme the GUI uses, and the one
the companion study exists to validate — so exactly one factor varies here.

Candidates, all on the per-margin sample of a posterior draw — the per-state
sample ``{y_n : x_n = k}`` for state margins, and for the pair margins f_ij
of a general PMC (DerrodePieczynski_CSDA2013 Eqs. 12–14) the dual view
``{y_n : (x_n, x_{n+1}) = (i, j)} ∪
{y_{n+1} : (x_n, x_{n+1}) = (j, i)}`` (:func:`pmcprg.diagnostics.margin_sample`):

``mks``     the incumbent: ``pmcprg.diagnostics.mks_1samp``, max deviation of
            the joint empirical CDF. The only one that generalises to d > 1
            as written — and the most expensive by two orders of magnitude
            (16 ms against 0.1 ms at n = 240), for something that in d = 1
            measures very nearly what ``ks`` measures: 0.0786 against 0.0828
            on the same sample, the gap being the evaluation convention.
``ks``      the classical one-sample KS on the probability integral
            transform ``u = F_k(y)``, which under H₀ is uniform.
``cvm``     Cramér-von Mises on the same transform: ``1/(12n) + Σ(u_(i) −
            (2i−1)/2n)²``. Weights the whole support evenly.
``ad``      Anderson-Darling, the obvious competitor the audit did not
            name: the same idea divided by ``u(1−u)``, which is precisely
            the variance of the empirical process and puts the weight in the
            tails — where ``student`` and ``skew`` differ and nowhere else.
``energy``  two-sample energy distance against a deterministic quantile
            representation of the fitted margin (Székely-Rizzo).
``mmd``     maximum mean discrepancy, Gaussian kernel, bandwidth fixed once
            per state from the reference so the statistic stays comparable
            across replicates.
``v1``…``v4`` Neyman smooth-test components: the transform projected on
            orthonormal Legendre polynomials, reported squared. Each is
            asymptotically ``χ²₁`` under H₀ and each targets one deformation
            — ``v1`` location, ``v2`` dispersion, ``v3`` asymmetry, ``v4``
            tail weight.

            The rationale needs stating carefully, because the obvious one is
            wrong. It is *not* that a component has a tighter null: measured
            on i.i.d. uniforms, ``V²`` has an interquartile range 2.68 times
            its median, against 0.95 for Anderson-Darling — nearly three
            times *wider*. Relative dispersion compares statistics within the
            EDF family; it does not transfer across families, and using it
            here would be a mistake.

            What governs power is the non-centrality: the H₁ shift measured
            in units of the H₀ standard deviation. A component puts the
            entire deviation of one kind into a single degree of freedom,
            where CvM and AD dilute it across an integral over the whole
            support. Concentration of signal, not tightness of null.
``n4``      Neyman's omnibus ``N₄ = Σ_{j≤4} V_j²``, asymptotically ``χ²₄``.

Which component rejects is itself the diagnosis — a single p-value says a
margin is wrong, ``v3`` against ``v4`` says *how*.

The reference sample for ``energy`` and ``mmd`` is built from ``ppf`` on a
regular quantile grid rather than by drawing: a deterministic reference
carries no Monte-Carlo noise of its own, which keeps the null distribution
about the data instead of about the reference.

Usage
-----
    python report/margin_statistic_panel.py --reps 200 --B 200 --workers 8
    python report/margin_statistic_panel.py --report-only
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

from mks_specification_study import CONFIGS, build_models, wilson  # noqa: E402

OUT_DIR  = REPORT_DIR / "out" / "margin_statistic_panel"
OUT_FILE = OUT_DIR / "results.jsonl"

STATS     = ["mks", "ks", "cvm", "ad", "energy", "mmd",
             "v1", "v2", "v3", "v4", "n4"]
ALPHA     = 0.05
N_DEFAULT = 600
M_REF     = 800          # quantile grid for the energy / MMD reference


# ---------------------------------------------------------------------------
# The statistics
# ---------------------------------------------------------------------------

def _reference(margin, m: int = M_REF):
    """Deterministic quantile representation of a fitted margin.

    ``ppf((i + ½)/m)`` rather than ``rvs``: a reference that carries no
    randomness of its own keeps the null distribution about the data.
    Returns the grid and the median pairwise distance, which fixes the MMD
    bandwidth once and for all — recomputing it per replicate would make the
    statistic incomparable between replicates, and the null incoherent.
    """
    import numpy as np

    # `_MarginDist.ppf` is scalar-only — the package vectorises `cdf` and
    # `pdf` but not `ppf` — so this loops. It runs once per state per
    # dataset, not per replicate, so the cost does not matter here.
    q = (np.arange(m) + 0.5) / m
    y = np.asarray([margin.ppf(float(qq)) for qq in q], dtype=float).ravel()
    y = y[np.isfinite(y)]
    sub = y[:: max(1, y.size // 200)]                 # median heuristic, cheap
    d   = np.abs(sub[:, None] - sub[None, :])
    sigma = float(np.median(d[d > 0])) if np.any(d > 0) else 1.0
    return y, max(sigma, 1e-9)


def _statistics(sample, margin, ref, sigma, want):
    """All requested statistics for one per-state sample."""
    import numpy as np

    from pmcprg.diagnostics import mks_1samp

    x = np.asarray(sample, dtype=float).ravel()
    n = x.size
    out: dict[str, float] = {}

    # Probability integral transform: uniform under H₀.
    u = np.clip(np.asarray(margin.cdf_vec(x), dtype=float), 1e-12, 1 - 1e-12)
    us = np.sort(u)
    i  = np.arange(1, n + 1)

    if "ks" in want:
        out["ks"] = float(np.max(np.abs(
            np.maximum(i / n - us, us - (i - 1) / n))))
    if "cvm" in want:
        out["cvm"] = float(1.0 / (12 * n) + np.sum((us - (2 * i - 1) / (2 * n)) ** 2))
    if "ad" in want:
        # Anderson & Darling (1952), Ann. Math. Statist. 23(2), 193–212;
        # (1954), JASA 49(268), 765–769 — the A² computing formula.
        out["ad"] = float(-n - np.mean(
            (2 * i - 1) * (np.log(us) + np.log1p(-us[::-1]))))
    if any(s in want for s in ("v1", "v2", "v3", "v4", "n4")):
        # Orthonormal Legendre polynomials on [0, 1]: E[h_j(U)] = 0 and
        # Var[h_j(U)] = 1 for U uniform, so V_j = n^{-1/2} Σ h_j(u_i) is
        # asymptotically N(0, 1) under H₀ and V_j² is χ²₁.
        z = 2.0 * u - 1.0
        h = {
            "v1": math.sqrt(3.0) * z,
            "v2": math.sqrt(5.0) * 0.5 * (3.0 * z ** 2 - 1.0),
            "v3": math.sqrt(7.0) * 0.5 * (5.0 * z ** 3 - 3.0 * z),
            "v4": 3.0 * 0.125 * (35.0 * z ** 4 - 30.0 * z ** 2 + 3.0),
        }
        comp = {name: float(np.sum(vals) / math.sqrt(n))
                for name, vals in h.items()}
        for name, v in comp.items():
            if name in want:
                out[name] = v * v
        if "n4" in want:
            out["n4"] = float(sum(v * v for v in comp.values()))
    if "mks" in want:
        try:
            arr = x.reshape(-1, 1)

            def _cdf(point):
                return float(margin.cdf_vec(
                    np.atleast_1d(np.asarray(point, dtype=float)))[0])
            out["mks"] = float(mks_1samp(arr, cdf=_cdf, alpha=ALPHA).statistic)
        except Exception:
            out["mks"] = float("nan")
    if "energy" in want or "mmd" in want:
        dxy = np.abs(x[:, None] - ref[None, :])
        dxx = np.abs(x[:, None] - x[None, :])
        # Both statistics below drop their reference-reference term. It is a
        # constant of the state, so it shifts the observed value and every
        # null draw by the same amount and cannot move a bootstrap p-value —
        # but it does mean these are the statistics up to an additive
        # constant, which is why `mmd` comes out negative. Do not read their
        # magnitudes as distances.
        if "energy" in want:
            # Székely & Rizzo (2013), J. Statist. Plann. Inference 143(8),
            # 1249–1272 — energy distance, minus its reference-reference term.
            out["energy"] = float(2.0 * dxy.mean() - dxx.mean())
        if "mmd" in want:
            # Gretton et al. (2012), JMLR 13, 723–773 — unbiased MMD² with a
            # Gaussian kernel, minus its reference-reference term.
            kxy = np.exp(-(dxy ** 2) / (2 * sigma ** 2))
            kxx = np.exp(-(dxx ** 2) / (2 * sigma ** 2))
            np.fill_diagonal(kxx, 0.0)
            out["mmd"] = float(kxx.sum() / (n * (n - 1)) - 2.0 * kxy.mean())
    return out


# ---------------------------------------------------------------------------
# One dataset
# ---------------------------------------------------------------------------

def _row_key(key) -> dict:
    """``{"state": k}`` for a state margin, ``{"pair": [i, j]}`` for f_ij."""
    return {"pair": list(key)} if isinstance(key, tuple) else {"state": key}


def _draw_stats(mdl, Yv, rng, refs, want):
    """Per-margin statistics on one FFBS posterior draw.

    Keys are the states for state margins and the pairs ``(i, j)`` for pair
    margins (:func:`pmcprg.diagnostics.margin_keys`).
    """
    from pmcprg.diagnostics   import margin_keys, margin_of, margin_sample
    from pmcprg.pmc.inference import (
        forward, precompute_weights, sample_posterior,
    )

    W, f_pdf = precompute_weights(mdl, Yv)
    alpha, _ = forward(mdl, Yv, W=W, f_pdf=f_pdf)
    lab = sample_posterior(mdl, Yv, rng, W=W, f_pdf=f_pdf, alpha_hat=alpha)

    out = {}
    for key in margin_keys(mdl):
        sel = margin_sample(Yv, lab, key)
        if len(sel) < 20:
            out[key] = ({s: float("nan") for s in want}, len(sel))
            continue
        ref, sigma = refs[key]
        out[key] = (_statistics(sel, margin_of(mdl, key), ref, sigma, want),
                    len(sel))
    return out


def run_one(task) -> dict:
    import numpy as np

    from pmcprg.diagnostics  import margin_keys, margin_of
    from pmcprg.pmc.simulate import simulate

    config, N, rep, B = task
    gen, ref_model = build_models(config)
    want = set(STATS)

    refs = {k: _reference(margin_of(ref_model, k)) for k in margin_keys(ref_model)}

    t0 = time.perf_counter()
    _, Y = simulate(gen, N=N, seed=700_000 + 1000 * rep + N)
    rng = np.random.default_rng(rep)
    observed = _draw_stats(ref_model, Y, rng, refs, want)

    null: dict = {(k, s): [] for k in observed for s in want}
    for _ in range(B):
        try:
            _, Yb = simulate(ref_model, N=N,
                             seed=int(rng.integers(0, 2**31 - 1)))
            for k, (vals, _m) in _draw_stats(ref_model, Yb, rng,
                                             refs, want).items():
                for s, v in vals.items():
                    if np.isfinite(v):
                        null[(k, s)].append(v)
        except Exception:
            pass

    rows = []
    for k, (vals, m) in observed.items():
        for s, v in vals.items():
            draws = np.asarray(null[(k, s)], dtype=float)
            rows.append({
                **_row_key(k), "statistic": s, "n": int(m),
                "value": (float(v) if np.isfinite(v) else None),
                "p": (float(np.mean(draws >= v))
                      if draws.size and np.isfinite(v) else None),
                "n_valid": int(draws.size),
            })
    return {"config": config, "N": N, "rep": rep, "B": B,
            "seconds": round(time.perf_counter() - t0, 2), "rows": rows}


# ---------------------------------------------------------------------------

def summarise(rows_all) -> None:
    import numpy as np

    acc: dict = {}
    for row in rows_all:
        for r in row["rows"]:
            if r["p"] is None:
                continue
            acc.setdefault((row["config"], r["statistic"]), []).append(r["p"])

    print(f"\nRejets au seuil {ALPHA:.0%} — toutes les statistiques sont "
          f"calibrées par le MÊME bootstrap,\ndonc `h0` doit valoir ~{ALPHA} "
          f"partout et seules les autres lignes comparent la puissance.")
    print(f"\n{'config':<10}" + "".join(f"{s:>22}" for s in STATS))
    print("-" * (10 + 22 * len(STATS)))
    for config in CONFIGS:
        cells = []
        for s in STATS:
            ps = acc.get((config, s))
            if not ps:
                cells.append(f"{'—':>22}")
                continue
            p = np.asarray(ps)
            k5 = int((p < ALPHA).sum())
            lo, hi = wilson(k5, p.size)
            cells.append(f"{k5 / p.size:>8.3f} [{lo:.2f},{hi:.2f}]")
        print(f"{config:<10}" + "".join(f"{c:>22}" for c in cells))
    n = next((len(v) for v in acc.values()), 0)
    print(f"\n  n = {n} mesures par cellule (jeux × états)")


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
    tasks = [(c, args.N, rep, args.B) for rep in range(args.reps)
             for c in wanted if (c, args.N, rep, args.B) not in done]
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


if __name__ == "__main__":
    main()
