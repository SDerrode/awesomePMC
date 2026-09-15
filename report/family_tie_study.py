"""How often is the copula family ICE selects statistically tied with the runner-up?

Audit FR-6. On Exp. 3 settings of the CSDA-2013 reproduction
(``report/reproduce_csda2013.py``, ``run_exp3``: same truth, candidates,
margins, prior, N = 2500, ``max_iter`` = 30, initial model and seeds
``3000 + r``), each ICE run is followed by
:func:`pmcprg.diagnostics.model_selection.ice_pair_comparisons` on the fitted
model: for every pair of states, the family the criterion selects is tested
against the runner-up on the ξ-weighted pseudo-observations.

Reported per (setting, criterion), over runs × pairs:

* ``hit``        — selected family = true family;
* ``tie``        — Vuong at 5 %, HAC (Newey–West 1994 bandwidth), n_eff = Σξ,
                   with the correction consistent with the criterion
                   (mle → none, aic → Akaike, bic → Schwarz, others → none);
* ``tie_none``   — the same without correction;
* ``tie_iid``    — the consistent correction with the i.i.d. variance
                   (bandwidth 0), to see what the HAC changes;
* ``tie_clarke`` — Clarke's sign test, HAC, consistent correction (read the
                   caveat in ``pmcprg.diagnostics.model_selection``: it ignores the
                   estimation of θ̂ and over-rejects for close families);
* ``|set|``      — size of the set of families not significantly worse than
                   the selected one (Vuong, consistent correction);
* ``true∈set``   — the true family is in that set.

Nothing is written to ``report/results``: the tables are printed, and
``--out FILE.csv`` saves the per-(run, pair) rows wherever you choose.

Usage
-----
::

    PYTHONPATH=. python report/family_tie_study.py \
        --configs exp2_orig_p exp1_orig_p --criteria mle aic bic huard \
        --runs 6 --jobs 2 [--out /tmp/ties.csv]
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import logging
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("repro_csda2013", _HERE / "reproduce_csda2013.py")
R = importlib.util.module_from_spec(_spec)
sys.modules["repro_csda2013"] = R
_spec.loader.exec_module(R)

from pmcprg.diagnostics.model_selection import (  # noqa: E402
    CRITERION_CORRECTION, clarke_test, vuong_test, ice_pair_comparisons,
)
from pmcprg.pmc import ice, simulate  # noqa: E402

logging.getLogger("pmcprg").setLevel(logging.ERROR)

P_ORIG = [[0.50, 0.05], [0.05, 0.40]]
P_BAL = [[0.35, 0.15], [0.15, 0.35]]
TRUTH_1 = {(0, 0): ("c1", 0.7), (0, 1): ("c3", 0.4), (1, 0): ("c3", 0.4), (1, 1): ("c6", 0.7)}
TRUTH_2 = {(0, 0): ("c2", 0.25), (0, 1): ("c5", 0.10), (1, 0): ("c5", 0.10), (1, 1): ("c6", 0.20)}
# The four base configurations of ``run_exp3`` (state margins).
CONFIGS = {
    "exp1_orig_p": (["c1", "c3", "c6"], TRUTH_1, "gauss", P_ORIG),
    "exp2_orig_p": (["c2", "c3", "c5", "c6"], TRUTH_2, "gamma", P_ORIG),
    "exp1_bal_p": (["c1", "c3", "c6"], TRUTH_1, "gauss", P_BAL),
    "exp2_bal_p": (["c2", "c3", "c5", "c6"], TRUTH_2, "gamma", P_BAL),
}
N_OBS, MAX_ITER, ALPHA = 2500, 30, 0.05


def _run(args: tuple) -> list[dict]:
    label, criterion, seed = args
    cands, truth, margin_kind, p = CONFIGS[label]
    margins = R.MARGIN_SETS["state"][margin_kind]
    cand_short = [R.COPULA_REGISTRY[c]["short"] for c in cands]
    init = {(i, j): (cands[0], 0.0) for i in range(2) for j in range(2)}
    truth_mdl = R._build_pmc_model_per_pair(truth, margins, p)
    init_mdl = R._build_pmc_model_per_pair(init, margins, p)
    _, Y = simulate(truth_mdl, N=N_OBS, seed=seed)
    fitted, trace = ice(init_mdl, Y, ice_cfg={
        "max_iter": MAX_ITER, "candidates": cand_short,
        "fit_margins": False, "selection_criterion": criterion,
    })
    correction = CRITERION_CORRECTION.get(criterion, "none")
    res = ice_pair_comparisons(fitted, Y, candidates=cand_short, criterion=criterion,
                               alpha=ALPHA, correction=correction)
    rows = []
    for (i, j), pc in res.items():
        true_name = R.COPULA_REGISTRY[truth[(i, j)][0]]["short"]
        row = {
            "config": label, "criterion": criterion, "seed": seed, "i": i, "j": j,
            "n_iter": trace.n_iters, "sum_w": round(pc.sum_weights, 2),
            "true": true_name, "model_family": pc.model_family,
            "selected": pc.selected, "runner_up": pc.runner_up,
            "hit": int(pc.selected == true_name),
            "gap": (pc.scores[pc.selected] - pc.scores[pc.runner_up]) if pc.runner_up else np.nan,
        }
        if pc.vs_runner_up is None:
            rows.append(row)
            continue
        a, b = pc.selected, pc.runner_up
        la, lb, w = pc.log_densities[a], pc.log_densities[b], pc.weights
        kw = dict(k_a=pc.n_params[a], k_b=pc.n_params[b], alpha=ALPHA)
        v_none = vuong_test(la, lb, w, **kw)
        v_iid = vuong_test(la, lb, w, correction=correction, bandwidth=0, **kw)
        c_hac = clarke_test(la, lb, w, correction=correction, **kw)
        vs = pc.vs_runner_up
        row.update({
            "z": vs.statistic, "p": vs.p_value, "L": vs.bandwidth, "decision": vs.decision,
            "tie": int(vs.decision != "A"), "tie_none": int(v_none.decision != "A"),
            "tie_iid": int(v_iid.decision != "A"), "tie_clarke": int(c_hac.decision != "A"),
            "set_size": len(pc.confidence_set.members),
            "true_in_set": int(true_name in pc.confidence_set.members),
            "set": "|".join(pc.confidence_set.members),
        })
        rows.append(row)
    return rows


def _summary(rows: list[dict]) -> None:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["config"], r["criterion"])].append(r)
    head = (f"{'setting':12s} {'crit':6s} {'cells':>5s} {'hit':>5s} {'tie':>5s} "
            f"{'none':>5s} {'iid':>5s} {'clarke':>6s} {'|set|':>5s} {'true∈set':>8s} "
            f"{'miss→tie':>8s} {'sel≠model':>9s}")
    print(head)
    print("-" * len(head))
    for (label, crit), rs in groups.items():
        rs = [r for r in rs if "tie" in r]
        n = len(rs)
        if not n:
            continue
        mean = lambda k: np.mean([r[k] for r in rs])  # noqa: E731
        misses = [r for r in rs if not r["hit"]]
        miss_tie = np.mean([r["tie"] for r in misses]) if misses else np.nan
        print(f"{label:12s} {crit:6s} {n:5d} {mean('hit'):5.2f} {mean('tie'):5.2f} "
              f"{mean('tie_none'):5.2f} {mean('tie_iid'):5.2f} {mean('tie_clarke'):6.2f} "
              f"{mean('set_size'):5.2f} {mean('true_in_set'):8.2f} {miss_tie:8.2f} "
              f"{np.mean([r['selected'] != r['model_family'] for r in rs]):9.2f}")
    print("\nPer pair (tie rate with the consistent correction; mean Σξ; hit rate):")
    per = defaultdict(list)
    for r in rows:
        if "tie" in r:
            per[(r["config"], r["criterion"], r["i"], r["j"])].append(r)
    for (label, crit, i, j), rs in per.items():
        print(f"  {label:12s} {crit:6s} ({i},{j}) true={rs[0]['true']:8s} "
              f"tie={np.mean([r['tie'] for r in rs]):.2f}  hit={np.mean([r['hit'] for r in rs]):.2f}  "
              f"Σξ={np.mean([r['sum_w'] for r in rs]):7.1f}  "
              f"selected={dict(sorted(_count(r['selected'] for r in rs).items()))}")


def _count(it) -> dict:
    out: dict = defaultdict(int)
    for x in it:
        out[x] += 1
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--configs", nargs="+", default=["exp2_orig_p"], choices=sorted(CONFIGS))
    ap.add_argument("--criteria", nargs="+", default=["mle", "aic", "bic"])
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    tasks = [(label, crit, 3000 + r) for label in a.configs for crit in a.criteria
             for r in range(a.runs)]
    t0 = time.time()
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for k, rs in enumerate(ex.map(_run, tasks, chunksize=1), 1):
            rows.extend(rs)
            print(f"  {k}/{len(tasks)} {tasks[k - 1]}  {time.time() - t0:.0f}s", flush=True)
    _summary(rows)
    if a.out is not None:
        keys = sorted({k for r in rows for k in r})
        with open(a.out, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=keys)
            wr.writeheader()
            wr.writerows(rows)
        print("rows ->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
