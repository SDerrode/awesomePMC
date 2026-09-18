#!/usr/bin/env python3
"""Why the ξ-weighted GoF test has no power — diagnosing the weighting.

Context
-------
``report/gof_power_study.py`` established, over 1200 datasets, that the S-6
GoF test holds its level but has essentially no power: against Clayton the
diagonal pairs reject at 0.100, against GH at 0.048 (nominal 0.05). Worse,
under GH at N = 1500 the pairs that reject are the **off-diagonal** ones —
independent by construction, τ = 0 — at 0.100, while the pairs that actually
carry the mis-modelled dependence sit at the nominal rate. The test sees the
misspecification and attributes it to the wrong cells.

The hypothesis that suggested itself: ξ is computed under the *model being
tested*. When that model is wrong, the posterior reweighting should
down-weight exactly the transitions the wrong family explains badly, moving
their mass — and the misfit with it — off the diagonal and onto the
low-weight pairs. If so, the weighting, not the statistic, is the culprit.

This script tests that, in two steps.

Probe A — is the mass actually displaced?
    Compare ξ computed under the generating model with ξ computed under the
    (wrong) Gaussian model, on the same data. Two predictions: total mass
    shifts from the diagonal to the off-diagonal, and at transition level the
    weight change on a diagonal pair is *negatively* correlated with how
    badly the Gaussian copula explains that transition.

Probe B — is the displacement the CAUSE?
    Recompute the same statistic under four weightings and compare the H1/H0
    separation of the medians, with bootstrap intervals:
      ``xi_wrong``  — ξ under the model being tested (what ships today);
      ``xi_oracle`` — ξ under the model that generated the data. Not
                      implementable in practice; it isolates the cause;
      ``xi_refit``  — ξ under a model whose pair copulas have been refitted
                      from the data (one ICE M-step, best of the candidate
                      families). This one IS implementable: it asks the
                      posterior to stop assuming the family under test;
      ``hard``      — hard assignment, argmax ξ under the tested model;
      ``uniform``   — no weighting at all.
    If the oracle restores the separation, the weighting is to blame. If it
    does not, the statistic is blind on its own.

Probe C — the paired comparison
    The five weightings run on the *same* datasets, so treating them as
    independent samples throws the pairing away and inflates every interval.
    Probe C computes, dataset by dataset, r = stat(scheme)/stat(xi_wrong),
    and reports the gain r(H1)/r(H0): above 1 the scheme amplifies the misfit
    more than it amplifies the noise. This is the comparison that actually
    settles whether a scheme is an improvement, and it is what showed the
    refit to be a net loss.

Usage
-----
    python report/gof_weighting_diagnosis.py            # R = 200, N = 1500
    python report/gof_weighting_diagnosis.py --reps 40  # a quick look
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

from pmcprg.diagnostics   import copula_pseudo_obs, margin_cdfs       # noqa: E402
from pmcprg.pmc.ice       import (                                    # noqa: E402
    _effective_n, _select_and_fit_copula, _weighted_cvm_statistic,
)
from pmcprg.pmc.inference import (                                      # noqa: E402
    backward, forward, joint_posteriors, precompute_weights,
)
from pmcprg.pmc.model    import PMCModel                                # noqa: E402
from pmcprg.pmc.simulate import simulate                                # noqa: E402

BASE_MODEL = REPO_ROOT / "pmcprg" / "pmc" / "models" / "pmc_gauss_k2.toml"
SCHEMES    = ("xi_wrong", "xi_oracle", "xi_refit", "hard", "uniform")
# The candidate list the ICE default uses; `xi_refit` picks among these.
CANDIDATES = ["Gauss", "GH", "Clayton", "Frank", "Joe"]


def with_family(name: str) -> PMCModel:
    """The fixture with its *dependent* pair copulas swapped, τ untouched."""
    base = PMCModel(BASE_MODEL)
    raw  = base.raw
    for blk in raw["copulas"]:
        if abs(float(blk["tau"])) > 1e-9:        # leave the independent pairs
            blk["name"] = name
    return PMCModel.from_dict(raw)


def refit_model(mdl: PMCModel, Y: np.ndarray,
                candidates: list[str] | None = None) -> PMCModel:
    """One ICE M-step on the copulas alone: the family stops being assumed.

    The posterior that weights the test is the whole problem — computed under
    the model on trial, it sheds exactly the transitions that model explains
    badly. This refits each pair copula from the data (best of
    ``candidates`` by the default criterion, weighted by the posterior of the
    model on trial) and hands back a model whose ξ no longer presupposes the
    family we are about to test.
    """
    xi  = xi_of(mdl, Y)
    F   = pair_cdfs(mdl, Y)
    raw = mdl.raw
    for blk in raw.get("copulas", []):
        i, j = int(blk["i"]), int(blk["j"])
        uv, w = copula_pseudo_obs(F, xi, i, j)
        if w.sum() < 1e-12:
            continue
        try:
            blk.update(_select_and_fit_copula(
                list(candidates or CANDIDATES),
                uv[:, 0], uv[:, 1], w,
            ))
        except Exception:
            pass                       # keep the incumbent block on failure
    return PMCModel.from_dict(raw)


def xi_of(mdl: PMCModel, Y: np.ndarray) -> np.ndarray:
    """ξ_n(i, j) = P(X_n=i, X_{n+1}=j | Y) under ``mdl``."""
    W, f_pdf = precompute_weights(mdl, Y)
    alpha, _ = forward(mdl, Y, W=W, f_pdf=f_pdf)
    return joint_posteriors(alpha, W, backward(mdl, Y, W=W))


def pair_cdfs(mdl: PMCModel, Y: np.ndarray) -> np.ndarray:
    """``F[n, i, j] = F_ij(y_n)``.

    Pair margins of a general PMC (DerrodePieczynski_CSDA2013 Eqs. 12–14): the
    copula c_ij couples F_ij(y_n) with F_ji(y_{n+1}). For state margins F_ij =
    F_i whatever j and the array is the K state CDFs broadcast over j — the
    same floats the K-vector version produced.
    """
    return margin_cdfs(mdl, Y, clip=1e-12)


# ---------------------------------------------------------------------------
# Probe A — is the mass displaced?
# ---------------------------------------------------------------------------

def probe_a(reps: int, N: int) -> None:
    print("\n" + "=" * 78)
    print("Sonde A — déplacement de la masse postérieure (ξ faux / ξ vrai)")
    print("=" * 78)
    print(f"{'données':<10}{'masse diag':>14}{'masse hors-diag':>18}"
          f"{'corr(désajust., Δpoids)':>26}")
    gauss = with_family("Gauss")
    for fam in ("GH", "Clayton", "Gauss"):
        gen = with_family(fam)
        ratio_d, ratio_o, corrs = [], [], []
        for r in range(reps):
            _, Y = simulate(gen, N=N, seed=4000 + r)
            xi_t, xi_w = xi_of(gen, Y), xi_of(gauss, Y)
            K = gen.K
            dg = [(i, i) for i in range(K)]
            of = [(i, j) for i in range(K) for j in range(K) if i != j]
            ratio_d.append(sum(xi_w[:, i, j].sum() for i, j in dg)
                           / max(sum(xi_t[:, i, j].sum() for i, j in dg), 1e-12))
            ratio_o.append(sum(xi_w[:, i, j].sum() for i, j in of)
                           / max(sum(xi_t[:, i, j].sum() for i, j in of), 1e-12))
            # Transition-level: does the wrong model shed the misfitting ones?
            uv, _ = copula_pseudo_obs(pair_cdfs(gauss, Y), xi_w, 0, 0)
            misfit = (gen.copula(0, 0).logpdf_array(uv)
                      - gauss.copula(0, 0).logpdf_array(uv))
            dw = xi_w[:, 0, 0] - xi_t[:, 0, 0]
            ok = np.isfinite(misfit) & np.isfinite(dw)
            if ok.sum() > 10 and np.std(dw[ok]) > 0:
                corrs.append(float(np.corrcoef(misfit[ok], dw[ok])[0, 1]))
        se = lambda v: np.std(v) / max(np.sqrt(len(v)), 1)      # noqa: E731
        c  = np.asarray(corrs)
        c_txt = ("—  (Δ identiquement nul)" if c.size == 0
                 else f"{c.mean():+.4f} ± {se(c):.4f}")
        print(f"{fam:<10}{np.mean(ratio_d):>8.4f} ± {se(ratio_d):.4f}"
              f"{np.mean(ratio_o):>12.4f} ± {se(ratio_o):.4f}{c_txt:>26}")
    print("\n  Une masse hors-diagonale > 1 et une corrélation négative = le modèle")
    print("  erroné se débarrasse des transitions qu'il explique mal, au profit")
    print("  des paires de faible poids.")


# ---------------------------------------------------------------------------
# Probe B — is it the cause?
# ---------------------------------------------------------------------------

def probe_b(reps: int, N: int, seed: int = 0) -> None:
    gauss = with_family("Gauss")
    acc: dict[str, dict[str, dict[str, list]]] = {}
    paired: dict[str, dict] = {}
    for gen_name in ("Gauss", "GH", "Clayton"):
        gen = with_family(gen_name)
        slot = {s: {"diag": [], "off": []} for s in SCHEMES}
        per_dataset: dict = {}
        for r in range(reps):
            _, Y = simulate(gen, N=N, seed=6000 + r)
            xw, xo = xi_of(gauss, Y), xi_of(gen, Y)
            xr = xi_of(refit_model(gauss, Y), Y)
            F = pair_cdfs(gauss, Y)
            hard = xw.reshape(N - 1, -1).argmax(axis=1)
            K = gauss.K
            for i in range(K):
                for j in range(K):
                    uv, _ = copula_pseudo_obs(F, xw, i, j)
                    cop = gauss.copula(i, j)
                    weights = {
                        "xi_wrong":  xw[:, i, j],
                        "xi_oracle": xo[:, i, j],
                        "xi_refit":  xr[:, i, j],
                        "hard":      (hard == i * K + j).astype(float),
                        "uniform":   np.ones(N - 1),
                    }
                    key = "diag" if i == j else "off"
                    per = per_dataset.setdefault(key, {})
                    for s in SCHEMES:
                        w = np.asarray(weights[s], dtype=float)
                        if w.sum() < 1e-9 or _effective_n(w) < 4:
                            continue
                        try:
                            val = _weighted_cvm_statistic(uv, w, cop)
                        except Exception:
                            continue
                        slot[s][key].append(val)
                        # Paired bookkeeping: same dataset, same pair, every
                        # scheme — the five weightings are not independent
                        # samples and an unpaired interval throws that away.
                        per.setdefault(s, {}).setdefault(r, []).append(val)
        acc[gen_name] = slot
        paired[gen_name] = {
            key: {s: np.array([np.mean(v[r]) for r in sorted(v)])
                  for s, v in d.items()}
            for key, d in per_dataset.items()
        }

    rng = np.random.default_rng(seed)

    def ratio_ci(h1, h0, B: int = 2000):
        h1, h0 = np.asarray(h1), np.asarray(h0)
        r = float(np.median(h1) / np.median(h0))
        draws = [np.median(rng.choice(h1, h1.size))
                 / np.median(rng.choice(h0, h0.size)) for _ in range(B)]
        lo, hi = np.percentile(draws, [2.5, 97.5])
        return r, float(lo), float(hi)

    for key, title in (("diag", "paires DIAGONALES — porteuses de la dépendance"),
                       ("off",  "paires HORS-DIAGONALE — indépendantes, τ = 0")):
        print("\n" + "=" * 78)
        print(f"Sonde B — séparation H1/H0 des médianes, {title}")
        print("=" * 78)
        print(f"{'pondération':<12}{'GH / Gauss':>26}{'Clayton / Gauss':>26}")
        for s in SCHEMES:
            cells = []
            for h1 in ("GH", "Clayton"):
                r, lo, hi = ratio_ci(acc[h1][s][key], acc["Gauss"][s][key])
                cells.append(f"{r:.2f}  [{lo:.2f}, {hi:.2f}]")
            print(f"{s:<12}{cells[0]:>26}{cells[1]:>26}")
    # Paired comparison: does a scheme lift the statistic MORE under H1 than
    # under H0, dataset by dataset? Far more sensitive than two independent
    # intervals, because every scheme saw the same data.
    for key, title in (("diag", "DIAGONALES"), ("off", "HORS-DIAGONALE")):
        print("\n" + "=" * 78)
        print(f"Comparaison APPARIÉE vs xi_wrong — paires {title}")
        print("  ratio médian r = stat(schéma)/stat(xi_wrong) par jeu ;")
        print("  gain = r(H1)/r(H0). > 1 = le schéma amplifie le désajustement.")
        print("=" * 78)
        print(f"{'schéma':<12}{'GH : gain [IC]':>28}{'Clayton : gain [IC]':>28}")
        for s in SCHEMES:
            if s == "xi_wrong":
                continue
            cells = []
            for h1 in ("GH", "Clayton"):
                r0 = paired["Gauss"][key][s] / paired["Gauss"][key]["xi_wrong"]
                r1 = paired[h1][key][s] / paired[h1][key]["xi_wrong"]
                g  = float(np.median(r1) / np.median(r0))
                draws = [np.median(rng.choice(r1, r1.size))
                         / np.median(rng.choice(r0, r0.size)) for _ in range(2000)]
                lo, hi = np.percentile(draws, [2.5, 97.5])
                cells.append(f"{g:.3f}  [{lo:.3f}, {hi:.3f}]")
            print(f"{s:<12}{cells[0]:>28}{cells[1]:>28}")

    print("\n  Un intervalle contenant 1 = aucune séparation. L'oracle n'étant pas")
    print("  réalisable, il ne sert qu'à isoler la cause : s'il rétablit la")
    print("  séparation, c'est la pondération ; sinon, c'est la statistique.")
    print("  `xi_refit` est la version réalisable de l'oracle : si elle en")
    print("  approche, le correctif est à portée.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--N", type=int, default=1500)
    args = ap.parse_args()
    probe_a(args.reps, args.N)
    probe_b(args.reps, args.N)


if __name__ == "__main__":
    main()
