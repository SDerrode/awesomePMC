#!/usr/bin/env python3
"""Where does the dispersion of the null come from? (opportunity O7)

``report/margin_statistic_panel.py`` left a paradox. Under a Student-t
alternative, Cramér-von Mises and Anderson-Darling carry the **largest**
signal of the six candidates — median ratios of 1.77 and 1.82 against KS's
1.28 — and have **no power at all**, sitting at the nominal 0.045. The reason
was traced to the spread of their null: an interquartile range *larger than
the median*, so a 1.8× shift never clears the 95th percentile.

That much was measured. What caused the spread was not, and the hypothesis on
record was wrong. It read: *under serial dependence extreme values arrive in
runs, which should hurt tail-weighted statistics most*. This script separates
the candidate causes and shows that dependence is the smaller half of the
story.

Three sources, separated by construction
----------------------------------------
* **random sample size** — removed by truncating every per-state sample to a
  common ``n``, so size cannot be a factor;
* **copula dependence** between consecutive observations — varied through τ;
* **state persistence** — varied through ``a = P(X_{n+1} = X_n)``, which
  controls how much of the copula's serial dependence survives *inside* a
  per-state sample: with ``a`` near ½ consecutive observations rarely share a
  state, with ``a`` near 1 a per-state sample is long runs of coupled pairs.

Labels are the true ones throughout, so labelling is not a factor either.

Two controls
------------
* an **i.i.d. baseline** drawn straight from the margin, outside any model;
* the **theoretical** dispersion of each statistic, obtained by simulating it
  on i.i.d. uniforms with no model in sight. If the i.i.d. baseline matches
  the theory, the measurement itself is sound.

Usage
-----
    python report/null_dispersion_study.py --reps 400
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = REPORT_DIR.parent
for p in (str(REPO_ROOT), str(REPORT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from margin_statistic_panel import STATS, _reference, _statistics  # noqa: E402
from pmcprg.diagnostics  import margin_keys, margin_of, margin_sample  # noqa: E402
from pmcprg.pmc.model    import PMCModel                              # noqa: E402
from pmcprg.pmc.simulate import simulate                              # noqa: E402

BASE_MODEL = REPO_ROOT / "pmcprg" / "pmc" / "models" / "pmc_gauss_k2.toml"


def _model(tau: float, a: float) -> PMCModel:
    """The fixture with dependence and persistence set explicitly.

    ``tau`` goes on the diagonal pairs only — the off-diagonal ones are
    independent in this fixture and stay that way. ``a`` is written straight
    into the joint prior of a symmetric two-state chain.
    """
    raw = PMCModel(BASE_MODEL).raw
    for blk in raw["copulas"]:
        blk["name"] = "Gauss"
        blk["tau"]  = tau if blk["i"] == blk["j"] else 0.0
    raw["prior"]["p"] = [[a / 2, (1 - a) / 2], [(1 - a) / 2, a / 2]]
    return PMCModel.from_dict(raw)


def _dispersion(samples, margin, grid, sigma):
    """Interquartile range over the median, per statistic."""
    vals = {s: [] for s in STATS}
    for x in samples:
        v = _statistics(x, margin, grid, sigma, set(STATS))
        for s in STATS:
            if np.isfinite(v[s]):
                vals[s].append(v[s])
    out = {}
    for s in STATS:
        arr = np.asarray(vals[s])
        med = np.median(arr)
        out[s] = float((np.percentile(arr, 75) - np.percentile(arr, 25))
                       / abs(med)) if med else float("nan")
    return out


def _theoretical(n: int, reps: int = 20_000, seed: int = 0):
    """Dispersion of the EDF statistics on i.i.d. uniforms — no model at all.

    The control that tells whether the rest of the table is measuring the
    statistics or the harness.
    """
    rng = np.random.default_rng(seed)
    i = np.arange(1, n + 1)
    ks, cvm, ad = [], [], []
    for _ in range(reps):
        us = np.sort(rng.random(n))
        ks.append(np.max(np.maximum(i / n - us, us - (i - 1) / n)))
        cvm.append(1 / (12 * n) + np.sum((us - (2 * i - 1) / (2 * n)) ** 2))
        ad.append(-n - np.mean((2 * i - 1)
                               * (np.log(us) + np.log1p(-us[::-1]))))
    out = {}
    for name, arr in (("ks", ks), ("cvm", cvm), ("ad", ad)):
        a = np.asarray(arr)
        out[name] = float((np.percentile(a, 75) - np.percentile(a, 25))
                          / np.median(a))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--n", type=int, default=200, help="fixed per-state size")
    ap.add_argument("--N", type=int, default=900, help="series length")
    args = ap.parse_args()

    ref = PMCModel(BASE_MODEL)
    # State 0 for state margins; the pair (0, 0) — sample in the dual view —
    # for the pair margins of a general PMC (DerrodePieczynski_CSDA2013 Eqs. 12–14).
    key = margin_keys(ref)[0]
    margin = margin_of(ref, key)
    grid, sigma = _reference(margin)
    rng = np.random.default_rng(0)

    conds: dict[str, list] = {}
    conds["i.i.d."] = [
        np.asarray([margin.ppf(float(q)) for q in rng.random(args.n)])
        for _ in range(args.reps)
    ]
    for tau, a in ((0.0, 0.5), (0.0, 0.9), (0.6, 0.5), (0.6, 0.9), (0.6, 0.97)):
        mdl, kept = _model(tau, a), []
        for r in range(args.reps):
            X, Y = simulate(mdl, N=args.N, seed=20_000 + r)
            sel = margin_sample(Y, X, key)
            if len(sel) >= args.n:
                kept.append(sel[: args.n])       # fixed n: size is not a factor
        conds[f"τ={tau}  a={a}"] = kept

    print(f"Dispersion de la nulle (IQR / médiane), n fixé à {args.n}, "
          f"R={args.reps}, étiquettes vraies")
    print(f"{'condition':<16}{'jeux':>7}" + "".join(f"{s:>9}" for s in STATS))
    print("-" * (23 + 9 * len(STATS)))
    for name, samples in conds.items():
        d = _dispersion(samples, margin, grid, sigma)
        print(f"{name:<16}{len(samples):>7}"
              + "".join(f"{d[s]:>9.3f}" for s in STATS))

    th = _theoretical(args.n)
    print("\nContrôle théorique (uniformes i.i.d., aucun modèle) : "
          + "  ".join(f"{k}={v:.3f}" for k, v in th.items()))
    print("  Si la ligne « i.i.d. » ci-dessus le rejoint, la mesure porte bien")
    print("  sur les statistiques et non sur le harnais.")


if __name__ == "__main__":
    main()
