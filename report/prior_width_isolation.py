"""Prior-width vs sparsity: same data, same truth, only the Huard support varies.

Usage:  python report/prior_width_isolation.py OUT.csv [JOBS]   (JOBS default 3)

huard        -> uniform prior on each family's own tau-range (CSDA-2013 Eq. 20)
huard_common -> uniform prior on the intersection of all candidates' ranges
mle          -> reference, no prior at all
"""
import csv
import importlib.util
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

spec = importlib.util.spec_from_file_location("repro", "report/reproduce_csda2013.py")
m = importlib.util.module_from_spec(spec); sys.modules["repro"] = m; spec.loader.exec_module(m)

RUNS = 100
# Worker processes: second CLI argument, default 3. It was hard-wired to 8,
# which saturated the machine for the whole campaign.
JOBS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
p_orig = [[0.50, 0.05], [0.05, 0.40]]
p_bal  = [[0.35, 0.15], [0.15, 0.35]]
CFGS = [
    ("exp1_orig_p", ["c1","c3","c6"], {(0,0):("c1",0.7),(0,1):("c3",0.4),(1,0):("c3",0.4),(1,1):("c6",0.7)}, m.GAUSSIAN_MARGINS_K2, p_orig),
    ("exp2_orig_p", ["c2","c3","c5","c6"], {(0,0):("c2",0.25),(0,1):("c5",0.10),(1,0):("c5",0.10),(1,1):("c6",0.20)}, m.GAMMA_MARGINS_K2, p_orig),
    ("exp1_bal_p",  ["c1","c3","c6"], {(0,0):("c1",0.7),(0,1):("c3",0.4),(1,0):("c3",0.4),(1,1):("c6",0.7)}, m.GAUSSIAN_MARGINS_K2, p_bal),
    ("exp2_bal_p",  ["c2","c3","c5","c6"], {(0,0):("c2",0.25),(0,1):("c5",0.10),(1,0):("c5",0.10),(1,1):("c6",0.20)}, m.GAMMA_MARGINS_K2, p_bal),
]
CRITS = ["mle", "huard", "huard_common", "huard_global"]


def main(out_path):
    out = pathlib.Path(out_path)
    fh = open(out, "w", newline="")
    w = csv.DictWriter(fh, fieldnames=["config","criterion","run","i","j","true","sel","hit"])
    w.writeheader()
    with ProcessPoolExecutor(max_workers=JOBS) as ex:
        for label, cands, truth, margins, p in CFGS:
            short = {k: m.COPULA_REGISTRY[v[0]]["short"] for k, v in truth.items()}
            cand_short = [m.COPULA_REGISTRY[c]["short"] for c in cands]
            init = {(i, j): (cands[0], 0.0) for i in range(2) for j in range(2)}
            for crit in CRITS:
                args = [(truth, cand_short, init, margins, p, 2500, 30, crit, 3000 + r)
                        for r in range(RUNS)]
                for r, res in enumerate(ex.map(m._exp3_run_worker, args, chunksize=1)):
                    for (i, j), (sel, tau) in res["selected"].items():
                        w.writerow({"config": label, "criterion": crit, "run": r,
                                    "i": i, "j": j, "true": short[(i, j)],
                                    "sel": sel, "hit": int(sel == short[(i, j)])})
                fh.flush()                      # survive a crash mid-campaign
                print(f"  {label} / {crit} : termine", flush=True)
    fh.close()
    print("ecrit ->", out)


if __name__ == "__main__":           # required: spawn re-imports this module
    main(sys.argv[1])
