#!/usr/bin/env python3
"""Which GoF statistic sees an upper tail? — a candidate panel (audit S-6).

Context
-------
``report/gof_power_study.py`` measured the shipped test over 1200 datasets:
level correct, power ~0.10 against Clayton and 0.048 (i.e. none) against GH.
``report/gof_weighting_diagnosis.py`` then showed *why*, with an oracle
experiment: for Clayton the ξ weighting absorbs part of the signal, but for
GH the weighting is innocent — even with the true-model posterior the
separation is 1.07 [0.89, 1.26], i.e. nothing. The ξ-weighted Cramér-von
Mises statistic simply does not see an upper tail.

That is not surprising once stated: CvM measures ∑ w (C_n − C_θ)² over the
data, and Gaussian vs Gumbel at matched τ differ mostly where u, v → 1, a
region carrying little probability mass and where the two CDFs are both
close to their common asymptote.

This script therefore replaces the *statistic* rather than the weights,
which the diagnosis established were already the best implementable choice.
Every candidate keeps the ξ weighting and the same pseudo-observations, so
the comparison isolates the statistic.

Candidates
----------
``cvm``        the shipped baseline: ∑ w̃ (C_n − C_θ)².
``ad``         Anderson-Darling style variance stabilisation,
               ∑ w̃ (C_n − C_θ)² / [C_θ(1 − C_θ)], which is exactly the
               "give the corners their due" fix.
``kendall``    Cramér-von Mises on the Kendall process: compare the
               distribution of C(U, V) rather than C itself. K_θ is obtained
               by Monte-Carlo from the fitted copula.
``tail``       squared discrepancy of the tail-dependence functions
               λ_L(u) = C(u,u)/u and λ_U(u) = (1−2u+C(u,u))/(1−u) on grids
               near 0 and 1 — aimed straight at the diagnosed blindness.
``rosenblatt`` Genest-Rémillard S_n^(B): E = h(V|U) should be uniform and
               independent of U under H₀; CvM against the independence
               copula.

Method
------
Same paired design that settled the weighting question: every statistic is
computed on the *same* datasets, and the verdict is the gain
r(H1)/r(H0) where r = S(candidate)/S(cvm) per dataset. Above 1 the candidate
amplifies the misfit more than it amplifies the noise. Bootstrap intervals;
an interval containing 1 means no improvement over the baseline.

Usage
-----
    python report/gof_statistic_panel.py --reps 100
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPORT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = REPORT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from pmcprg.copulas._fit import _empirical_copula, _normalised_weights  # noqa: E402
from pmcprg.diagnostics   import copula_pseudo_obs, margin_cdfs          # noqa: E402
from pmcprg.pmc.ice       import _effective_n                            # noqa: E402
from pmcprg.pmc.inference import (                                       # noqa: E402
    backward, forward, joint_posteriors, precompute_weights,
)
from pmcprg.pmc.model    import PMCModel                                 # noqa: E402
from pmcprg.pmc.simulate import simulate                                 # noqa: E402

BASE_MODEL = REPO_ROOT / "pmcprg" / "pmc" / "models" / "pmc_gauss_k2.toml"
STATS      = ("cvm", "ad", "kendall", "tail", "rosenblatt")

_EPS = 1e-9


# ---------------------------------------------------------------------------
# Shared plumbing (identical to gof_weighting_diagnosis.py)
# ---------------------------------------------------------------------------

def with_family(name: str) -> PMCModel:
    base = PMCModel(BASE_MODEL)
    raw  = base.raw
    for blk in raw["copulas"]:
        if abs(float(blk["tau"])) > 1e-9:
            blk["name"] = name
    return PMCModel.from_dict(raw)


def xi_of(mdl: PMCModel, Y: np.ndarray) -> np.ndarray:
    W, f_pdf = precompute_weights(mdl, Y)
    alpha, _ = forward(mdl, Y, W=W, f_pdf=f_pdf)
    return joint_posteriors(alpha, W, backward(mdl, Y, W=W))


def pair_cdfs(mdl: PMCModel, Y: np.ndarray) -> np.ndarray:
    """``F[n, i, j] = F_ij(y_n)`` — pair margins of a general PMC
    (DerrodePieczynski_CSDA2013 Eqs. 12–14), or the K state CDFs broadcast
    over ``j`` (same floats)."""
    return margin_cdfs(mdl, Y, clip=1e-12)


# ---------------------------------------------------------------------------
# The candidate statistics — all ξ-weighted, all on the same pseudo-obs
# ---------------------------------------------------------------------------

def stat_cvm(uv, w, cop, **_) -> float:
    """The shipped baseline."""
    Cn = _empirical_copula(uv, uv, w)
    return float(np.dot(_normalised_weights(w), (Cn - cop.cdf_array(uv)) ** 2))


def stat_ad(uv, w, cop, **_) -> float:
    """Variance-stabilised CvM — the corners stop being drowned by the bulk."""
    Cn = _empirical_copula(uv, uv, w)
    Ct = np.clip(cop.cdf_array(uv), _EPS, 1.0 - _EPS)
    return float(np.dot(_normalised_weights(w), (Cn - Ct) ** 2 / (Ct * (1.0 - Ct))))


def stat_kendall(uv, w, cop, *, rng=None, m_mc: int = 4000, **_) -> float:
    """CvM on the Kendall process — the law of C(U,V), not C itself.

    K_θ(t) = P(C_θ(U,V) ≤ t) has no closed form here, so it is estimated once
    per call by Monte-Carlo from the fitted copula. The empirical side uses
    the ξ-weighted empirical copula evaluated at the data.
    """
    rng = rng or np.random.default_rng(0)
    Wn  = _empirical_copula(uv, uv, w)                    # C_n at the data
    sample = cop.sample(n=m_mc, seed=int(rng.integers(0, 2**31 - 1)))
    Wt = cop.cdf_array(sample)
    grid = np.linspace(0.01, 0.99, 99)
    wn   = _normalised_weights(w)
    Kn   = np.array([float(np.sum(wn[Wn <= t])) for t in grid])
    Kt   = np.array([float(np.mean(Wt <= t)) for t in grid])
    return float(np.mean((Kn - Kt) ** 2))


def stat_tail(uv, w, cop, *, n_grid: int = 12, **_) -> float:
    """Squared discrepancy of λ_L(u) and λ_U(u) — aimed at the tails.

    The empirical side uses the ξ-weighted diagonal section C_n(u, u); the
    model side uses the fitted copula's own C_θ(u, u), so the comparison is
    between two functions of the same quantity and needs no extra fitting.
    """
    wn  = _normalised_weights(w)
    lo  = np.linspace(0.02, 0.20, n_grid)
    hi  = np.linspace(0.80, 0.98, n_grid)
    out = 0.0
    for grid, side in ((lo, "lower"), (hi, "upper")):
        diag = np.column_stack((grid, grid))
        Ct   = cop.cdf_array(diag)
        Cn   = np.array([
            float(np.sum(wn[(uv[:, 0] <= u) & (uv[:, 1] <= u)])) for u in grid
        ])
        if side == "lower":
            emp, th = Cn / grid, Ct / grid
        else:
            emp = (1.0 - 2.0 * grid + Cn) / (1.0 - grid)
            th  = (1.0 - 2.0 * grid + Ct) / (1.0 - grid)
        out += float(np.mean((np.clip(emp, 0, 1) - np.clip(th, 0, 1)) ** 2))
    return out


def stat_rosenblatt(uv, w, cop, **_) -> float:
    """Genest-Rémillard S_n^(B): E = h(V|U) ⟂ U and uniform under H₀.

    ``conditional_cdf`` is scalar-only in this package, so this is the
    expensive candidate; it is included because it is the one the literature
    recommends, and knowing its cost is part of the answer.
    """
    u = uv[:, 0]
    e = np.array([cop.conditional_cdf(float(v), float(uu))
                  for uu, v in zip(uv[:, 0], uv[:, 1], strict=False)])
    e = np.clip(e, _EPS, 1.0 - _EPS)
    ue = np.column_stack((u, e))
    Cn = _empirical_copula(ue, ue, w)
    return float(np.dot(_normalised_weights(w), (Cn - u * e) ** 2))


_FN = {"cvm": stat_cvm, "ad": stat_ad, "kendall": stat_kendall,
       "tail": stat_tail, "rosenblatt": stat_rosenblatt}


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--N", type=int, default=1500)
    ap.add_argument("--stats", default=",".join(STATS))
    args = ap.parse_args()
    stats = [s for s in args.stats.split(",") if s in _FN]

    gauss = with_family("Gauss")
    N     = args.N
    rng   = np.random.default_rng(0)
    # per[gen][key][stat] -> one value per dataset (mean over the pairs of
    # that kind), so the comparison can be paired.
    per: dict = {}

    for gen_name in ("Gauss", "GH", "Clayton"):
        gen = with_family(gen_name)
        acc: dict = {k: {s: {} for s in stats} for k in ("diag", "off")}
        for r in range(args.reps):
            _, Y = simulate(gen, N=N, seed=6000 + r)
            xw = xi_of(gauss, Y)
            F  = pair_cdfs(gauss, Y)
            K  = gauss.K
            for i in range(K):
                for j in range(K):
                    # (F_ij(y_n), F_ji(y_{n+1})) weighted by ξ_n(i, j).
                    uv, w = copula_pseudo_obs(F, xw, i, j)
                    if w.sum() < 1e-9 or _effective_n(w) < 4:
                        continue
                    cop = gauss.copula(i, j)
                    key = "diag" if i == j else "off"
                    for s in stats:
                        try:
                            val = _FN[s](uv, w, cop, rng=rng)
                        except Exception:
                            continue
                        if np.isfinite(val):
                            acc[key][s].setdefault(r, []).append(val)
            if (r + 1) % 20 == 0:
                print(f"  {gen_name}: {r + 1}/{args.reps}", flush=True)
        per[gen_name] = {
            k: {s: np.array([np.mean(v[rr]) for rr in sorted(v)])
                for s, v in d.items()}
            for k, d in acc.items()
        }

    def gain(h1: str, key: str, s: str, B: int = 2000):
        r0 = per["Gauss"][key][s] / per["Gauss"][key]["cvm"]
        r1 = per[h1][key][s]      / per[h1][key]["cvm"]
        g  = float(np.median(r1) / np.median(r0))
        draws = [np.median(rng.choice(r1, r1.size))
                 / np.median(rng.choice(r0, r0.size)) for _ in range(B)]
        lo, hi = np.percentile(draws, [2.5, 97.5])
        return g, float(lo), float(hi)

    for key, title in (("diag", "DIAGONALES — porteuses de la dépendance"),
                       ("off",  "HORS-DIAGONALE — indépendantes, τ = 0")):
        print("\n" + "=" * 78)
        print(f"Gain apparié vs `cvm` — paires {title}")
        print("  gain = r(H1)/r(H0), r = S(candidat)/S(cvm) par jeu.")
        print("  > 1 : le candidat amplifie le désajustement plus que le bruit.")
        print("=" * 78)
        print(f"{'statistique':<14}{'GH (queue haute)':>28}{'Clayton (queue basse)':>28}")
        for s in stats:
            if s == "cvm":
                continue
            cells = []
            for h1 in ("GH", "Clayton"):
                g, lo, hi = gain(h1, key, s)
                cells.append(f"{g:.3f}  [{lo:.3f}, {hi:.3f}]")
            print(f"{s:<14}{cells[0]:>28}{cells[1]:>28}")

    # The baseline's own separation, for scale.
    print("\nSéparation brute H1/H0 des médianes (rappel, paires diagonales) :")
    for s in stats:
        row = []
        for h1 in ("GH", "Clayton"):
            row.append(f"{h1}={np.median(per[h1]['diag'][s]) / np.median(per['Gauss']['diag'][s]):.2f}")
        print(f"  {s:<12} " + "  ".join(row))


if __name__ == "__main__":
    main()
