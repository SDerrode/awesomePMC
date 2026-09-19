"""
run_pamap2.py — state-dependent missingness on PAMAP2 (P6): description of the
real hand-IMU gaps, likelihood-ratio tests, classification and imputation
with the mechanisms ignorable / "state" / "state-markov".

Parts (``--part``, default all):

* ``describe`` — missing rate, bursts, onset and persistence of the real
  gaps per activity and group, at 100 Hz and on the 2 Hz windows (rules
  ``ge10``: ≥ 5 of 50 samples NaN, and ``any``); the mechanisms fitted from
  the true labels, label-based likelihood-ratio tests, and a parametric
  goodness-of-fit check of "state" and "state-markov" (masks redrawn by the
  library generators on the true class path).
* ``test`` — ``missingness_lr_test`` on the real series (real gaps only),
  HMC-IN K = 2 and 3 from the k-means start of the real-series study,
  "state" vs common, "state-markov" vs common, "state-markov" vs "state";
  the library's profile statistic (commit 0758e3a; the difference of the
  fits is kept as ``LR_fits``), χ² and a parametric bootstrap (``--boot`` B,
  rule ``ge10``; ``--boot-any`` for the rule ``any``; ``--boot-subjects``
  restricts it to some subjects). ``--test-pmc`` adds the K = 2 PMC with
  state margins (χ² only).
* ``real`` — classification of the real-gap series (rules ge10 and any):
  supervised oracles (HMC-IN and PMC pair margins, K = 2 and 3; mechanism
  from the labels) and unsupervised HMC-IN (ICE, 3 starts, mechanism
  estimated), under the three mechanisms. Each estimated mechanism is fitted
  twice: from the start itself (``start_from = "start"``) and from the
  ignorable fit of that start (``"ignorable fit"``, the protocol of the LR
  test). PMC pair K = 3: k-means start, ignorable fit and the two mechanisms
  estimated from it.
* ``controlled`` — the complete series with masks drawn by
  ``pmcprg.missing.patterns.state_dependent`` / ``state_markov`` on the true
  group path, parameters matched to the real rates of each rule; the same
  classifications (unsupervised: k-means start only) and the imputation of
  the missing values (``gaps.impute``) under the three mechanisms.

Every task is a pure function of its spec (seeds ``zlib.crc32`` of named
tuples; masks, FFBS draws and bootstrap streams never share a seed); results
go to ``results/*.csv``, finished tasks to a git-ignored cache
(``report/out/missing_state_pamap2/``; a rerun skips them). Run from the
repository root::

    OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python report/missing_state/pamap2/run_pamap2.py --jobs 4 \\
        --boot 99 --boot-any 0 --boot-subjects 108      # the committed results
    .venv/bin/python report/missing_state/pamap2/summarise.py
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import pickle  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pm_common as pc  # noqa: E402
import rs_common as rc  # noqa: E402

from pmcprg.missing import patterns  # noqa: E402
from pmcprg.pmc import classify, ice, impute  # noqa: E402
from pmcprg.pmc.missingness_lr import missingness_lr_test  # noqa: E402

RESULTS = HERE / "results"
GOF_SIMS = 200          # redrawn masks per goodness-of-fit cell (2 Hz)
GOF_SIMS_100HZ = 40     # redrawn masks at 100 Hz ("state-markov" per activity)
CONTROLLED_SEEDS = 5    # masks per controlled cell
IMPUTE_DRAWS = 200      # FFBS draws per imputation (CRPS)
IMPUTE_Q = (0.05, 0.5, 0.95)
ORACLE_KINDS = ("hmc_in", "pmc_pair")
TESTS = (("state", "common"), ("state-markov", "common"), ("state-markov", "state"))
#: The LR statistic of ``missingness_lr_test`` (part of the task specs, so
#: that the cache never mixes them): "profile" — profile log-likelihoods of
#: the mechanism at both fits' θ (library commit 0758e3a); the earlier
#: difference of the two ICE fixed points is kept in ``LR_fits``.
STATISTIC = "profile"
IMPUTATION_COLUMNS = ("setting", "subject", "K", "rule", "mask", "r", "M", "model", "fit",
                      "mechanism", "start_from", "subset", "n", "rmse", "mae", "crps", "cov90",
                      "width90", "impute_s")

logger = logging.getLogger("pamap2_missing_state")


# ---------------------------------------------------------------------------
# Part 1 — description of the real gaps
# ---------------------------------------------------------------------------

def _gof_stats(m: np.ndarray, block: int = 20) -> dict:
    starts, L = pc.bursts(m)
    n = (len(m) // block) * block
    cnt = m[:n].reshape(-1, block).sum(axis=1).astype(float)
    return {"M": float(m.sum()), "n_bursts": float(L.size),
            "mean_len": float(L.mean()) if L.size else 0.0,
            "max_len": float(L.max()) if L.size else 0.0,
            "share_in_runs": float(L[L >= 2].sum() / max(L.sum(), 1)),
            "dispersion": float(cnt.var(ddof=1) / cnt.mean()) if cnt.mean() > 0 else np.nan}


def _draw(mech: str, cls: np.ndarray, par: dict, seed: int) -> np.ndarray:
    Y0 = np.zeros(cls.size)
    if mech == "state":
        return patterns.state_dependent(Y0, cls, par["rates"], seed=seed)[1].astype(bool)
    return patterns.state_markov(Y0, cls, par["onset"], par["persistence"], seed=seed)[1].astype(bool)


def _gof(m: np.ndarray, cls: np.ndarray, n_classes: int, mech: str, S: int, key: tuple,
         block: int) -> list[dict]:
    par = pc.class_rates(m, cls, n_classes)
    obs = _gof_stats(m, block)
    sims = [_gof_stats(_draw(mech, cls, par, pc.seed("gof", *key, mech, s)), block) for s in range(S)]
    rows = []
    for stat, v in obs.items():
        sv = np.array([s[stat] for s in sims], float)
        sv = sv[np.isfinite(sv)]
        p_hi = (1 + np.sum(sv >= v)) / (sv.size + 1)
        p_lo = (1 + np.sum(sv <= v)) / (sv.size + 1)
        rows.append({"mechanism": mech, "statistic": stat, "observed": v,
                     "sim_median": float(np.median(sv)), "sim_lo": float(np.quantile(sv, 0.025)),
                     "sim_hi": float(np.quantile(sv, 0.975)),
                     "p_two_sided": float(min(1.0, 2 * min(p_hi, p_lo))), "S": int(sv.size)})
    return rows


def task_describe(spec: dict) -> dict:
    subj, data = spec["subject"], spec["data_dir"]
    quick = spec.get("quick", False)
    out = {"desc_100hz": [], "desc_windows": [], "gof": [], "sup_mech": [], "sup_lr": []}
    # --- 100 Hz: raw hand dropouts per activity and group ---
    act, m = pc.load_100hz(data, subj)
    levels = [("activity", act, sorted(set(np.unique(act).tolist())))]
    for K in pc.KS:
        levels.append((f"K{K}", pc.group_of_activity(act, K), [-1] + list(range(K))))
    for level, cls, codes in levels:
        for c in codes:
            r = pc.mask_stats(m, cls, c, hist_edges=(1, 2, 6, 21), block=100)
            out["desc_100hz"].append({"subject": subj, "level": level, "code": int(c), **r})
    # onset dispersion at 1 s and 10 s: bursts starting per block
    starts, _ = pc.bursts(m)
    onset = np.zeros(m.size, bool)
    onset[starts] = True
    for c in sorted(set(np.unique(act).tolist())):
        row = next(r for r in out["desc_100hz"] if r["level"] == "activity" and r["code"] == c)
        row["onset_disp_1s"] = pc.block_dispersion(onset, act, c, 100)
        row["onset_disp_10s"] = pc.block_dispersion(onset, act, c, 1000)
    # 100 Hz goodness of fit of "state-markov" per activity (onset dispersion)
    codes = np.unique(act)
    cls = np.searchsorted(codes, act)
    par = pc.class_rates(m, cls, codes.size)
    S = 4 if quick else GOF_SIMS_100HZ

    def stats100(mm):
        st, L = pc.bursts(mm)
        on = np.zeros(mm.size, bool)
        on[st] = True
        n1, n10 = (mm.size // 100) * 100, (mm.size // 1000) * 1000
        c1 = on[:n1].reshape(-1, 100).sum(axis=1).astype(float)
        c10 = on[:n10].reshape(-1, 1000).sum(axis=1).astype(float)
        return {"M": float(mm.sum()), "n_bursts": float(L.size), "mean_len": float(L.mean()),
                "max_len": float(L.max()), "share_len1": float(np.mean(L == 1)),
                "onset_disp_1s": float(c1.var(ddof=1) / c1.mean()),
                "onset_disp_10s": float(c10.var(ddof=1) / c10.mean())}

    obs = stats100(m)
    sims = [stats100(_draw("state-markov", cls, par, pc.seed("gof100", subj, s))) for s in range(S)]
    for stat, v in obs.items():
        sv = np.array([s[stat] for s in sims])
        out["gof"].append({"subject": subj, "resolution": "100Hz", "rule": "-", "classes": "activity",
                           "mechanism": "state-markov", "statistic": stat, "observed": v,
                           "sim_median": float(np.median(sv)), "sim_lo": float(np.quantile(sv, 0.025)),
                           "sim_hi": float(np.quantile(sv, 0.975)),
                           "p_two_sided": float(min(1.0, 2 * min((1 + np.sum(sv >= v)) / (S + 1),
                                                                 (1 + np.sum(sv <= v)) / (S + 1)))),
                           "S": S})
    # --- 2 Hz windows ---
    d = pc.windows(data, subj)
    lab = d["labels"]
    for rule, field in pc.RULES.items():
        mw = d[field].astype(bool)
        levels = [("activity", lab, sorted(set(np.unique(lab).tolist())))]
        for K in pc.KS:
            levels.append((f"K{K}", pc.groups(d, K), [-1] + list(range(K))))
        levels.append(("series", np.zeros(lab.size, int), [0]))
        for level, cls_w, codes_w in levels:
            for c in codes_w:
                r = pc.mask_stats(mw, cls_w, c, hist_edges=(1, 2, 3, 6), block=20)
                out["desc_windows"].append({"subject": subj, "rule": rule, "level": level,
                                            "code": int(c), **r})
        # goodness of fit of the two mechanisms, classes = K3 groups (+ transient) or activities
        g3 = pc.groups(d, 3)
        schemes = {"K3": (np.where(g3 < 0, 3, g3), 4)}
        codes_a = np.unique(lab)
        schemes["activity"] = (np.searchsorted(codes_a, lab), codes_a.size)
        for scheme, (cls_w, nc) in schemes.items():
            for mech in ("state", "state-markov"):
                for r in _gof(mw, cls_w, nc, mech, 20 if quick else GOF_SIMS,
                              (subj, rule, scheme), 20):
                    out["gof"].append({"subject": subj, "resolution": "2Hz", "rule": rule,
                                       "classes": scheme, **r})
        # supervised view: mechanisms from the labels, label-based LR tests
        for K in pc.KS:
            g = pc.groups(d, K)
            for mech in ("state", "state-markov"):
                for guard, c in (("guard", None), ("plain", 0.0)):
                    est = pc.mechanism_from_labels(mw, g, K, mech, pseudo_count=c)
                    out["sup_mech"].append({"subject": subj, "rule": rule, "K": K,
                                            "mechanism": mech, "estimate": guard,
                                            **pc.mechanism_params(est)})
            for r in pc.supervised_lr(mw, g, K):
                out["sup_lr"].append({"subject": subj, "rule": rule, "K": K, **r})
    return out


# ---------------------------------------------------------------------------
# Part 2 — likelihood-ratio tests on the real series
# ---------------------------------------------------------------------------

def _ice_cfg(kind: str, start: int) -> dict:
    """ICE settings of the real-series study (``rc.estimate``, ICE "available")."""
    return {"fit_margins": True, "candidates": list(pc.COPULA_CANDIDATES),
            "selection_criterion": "mle", "margin_selection_rule": rc.margin_rule(kind),
            "init": "model", "n_starts": 1, "missing_strategy": "available",
            "missing_seed": 1000 + start}


def _sorted_params(model) -> tuple[dict, np.ndarray, np.ndarray]:
    means = pc.state_means(model)
    order = np.argsort(means)
    mech = pc.permute_mechanism(model.missingness, order)
    return pc.mechanism_params(mech), means[order], order


def _alignment(model, Y, g, K) -> dict:
    """MPM of a fitted model vs the groups: Hungarian error and, per state
    (sorted by margin mean), the share of each group among its kept rows."""
    x, _, _ = classify(model, Y)
    keep = g >= 0
    means = pc.state_means(model)
    order = np.argsort(means)
    rank = np.empty(K, int)
    rank[order] = np.arange(K)
    xs = rank[x]
    relabel = pc.hungarian(g[keep], xs[keep], K)
    err = float(np.mean(relabel[xs[keep]] != g[keep]))
    comp = []
    for s in range(K):
        sel = keep & (xs == s)
        comp.append("/".join(f"{np.mean(g[sel] == k):.2f}" if sel.any() else "nan" for k in range(K)))
    return {"align_err": err, "state_composition": " | ".join(comp),
            "state_share": "|".join(f"{np.mean(xs == s):.3f}" for s in range(K))}


def task_test(spec: dict) -> dict:
    subj, K, rule, kind = spec["subject"], spec["K"], spec["rule"], spec["model"]
    alt, null, B = spec["alternative"], spec["null"], spec["B"]
    d = pc.windows(spec["data_dir"], subj)
    miss = d[pc.RULES[rule]].astype(bool)
    Y = d["Y_complete"].copy()
    Y[miss] = np.nan
    g = pc.groups(d, K)
    init, _ = rc.starting_model(kind, K, Y, 0, 1)
    cfg = _ice_cfg(kind, 0)
    t0 = time.perf_counter()
    res = missingness_lr_test(init, Y, alternative=alt, null=null, ice_cfg=cfg, n_bootstrap=B,
                              seed=pc.seed("boot", subj, K, rule, kind, alt, null))
    dt = time.perf_counter() - t0
    p_alt, means_alt, _ = _sorted_params(res.alt_model)
    p_null, means_null, _ = _sorted_params(res.null_model)
    boot = np.asarray(res.bootstrap_statistics, float)
    row = {"subject": subj, "K": K, "rule": rule, "model": kind, "alternative": alt, "null": null,
           "N": res.n_obs, "M": res.n_missing, "LR": res.statistic, "df": res.df,
           "p_chi2": res.p_value, "p_boot": res.p_value_bootstrap, "B": B,
           "B_valid": res.n_bootstrap_valid,
           "boot_q95": float(np.quantile(boot, 0.95)) if boot.size else np.nan,
           "boot_max": float(boot.max()) if boot.size else np.nan,
           "boot_mean": float(boot.mean()) if boot.size else np.nan,
           "ll_null": res.log_lik_null, "ll_alt": res.log_lik_alt,
           "LR_fits": getattr(res, "statistic_fits", res.statistic),
           "sup_ll_null": getattr(res, "sup_log_lik_null", res.log_lik_null),
           "sup_ll_alt": getattr(res, "sup_log_lik_alt", res.log_lik_alt),
           "profile_log_liks": json.dumps(getattr(res, "profile_log_liks", {})),
           "alt_rates": p_alt["rates"], "alt_onset": p_alt["onset"],
           "alt_persistence": p_alt["persistence"],
           "null_rates": p_null["rates"], "null_onset": p_null["onset"],
           "null_persistence": p_null["persistence"],
           "alt_means": "|".join(f"{v:.4f}" for v in means_alt),
           "null_means": "|".join(f"{v:.4f}" for v in means_null),
           "runtime_s": dt, **_alignment(res.alt_model, Y, g, K)}
    return {"tests": [row]}


# ---------------------------------------------------------------------------
# Part 3 — classification (and part 4 — imputation)
# ---------------------------------------------------------------------------

_ORACLE_CACHE: dict = {}


def oracle(data_dir: str, subj: int, K: int, kind: str):
    """Supervised oracle of the real-series study (complete series + labels)."""
    key = (subj, K, kind)
    if key not in _ORACLE_CACHE:
        d = pc.windows(data_dir, subj)
        _ORACLE_CACHE[key] = rc.supervised_model(kind, K, d["Y_complete"], pc.groups(d, K))
    return _ORACLE_CACHE[key]


def _by_activity(x, g, miss, labels) -> str:
    """MPM error per activity (no relabelling: oracles), at the missing and at
    the observed windows: ``id:err_missing:n_missing:err_observed:n_observed``
    joined by ``|``."""
    parts = []
    for a in sorted(set(np.unique(labels).tolist()) - {0}):
        sm, so = (labels == a) & miss, (labels == a) & ~miss
        em = float(np.mean(x[sm] != g[sm])) if sm.any() else float("nan")
        eo = float(np.mean(x[so] != g[so])) if so.any() else float("nan")
        parts.append(f"{a}:{em:.4f}:{int(sm.sum())}:{eo:.4f}:{int(so.sum())}")
    return "|".join(parts)


def _score(model, Y, g, K, miss, *, align, base, truth=None, impute_seed=None,
           labels=None) -> tuple[list, list]:
    t0 = time.perf_counter()
    x, gamma, ll = classify(model, Y)
    row = {**base, "ll": float(ll), "classify_s": time.perf_counter() - t0,
           **pc.classification_scores(g, gamma, K, miss, align=align),
           **{f"mech_{k}": v for k, v in pc.mechanism_params(model.missingness).items()}}
    if labels is not None:
        row["err_by_activity"] = _by_activity(x, g, miss, labels)
    imp_rows = []
    if truth is not None:
        t1 = time.perf_counter()
        imp = impute(model, Y, quantiles=IMPUTE_Q, n_samples=IMPUTE_DRAWS, rng=impute_seed)
        t_imp = time.perf_counter() - t1
        g3 = base.get("_g3")
        idx = imp.index
        subsets = {"all": np.ones(idx.size, bool)}
        if g3 is not None:
            for k, name in enumerate(pc.GROUP_NAMES[3]):
                subsets[name] = g3[idx] == k
            subsets["transient"] = g3[idx] < 0
        for name, sel in subsets.items():
            if sel.any():
                imp_rows.append({**{k: v for k, v in base.items() if not k.startswith("_")},
                                 "subset": name, **pc.imputation_scores(truth[idx], imp, sel),
                                 "impute_s": t_imp})
    row.pop("_g3", None)
    return [row], imp_rows


def _with_mech(model, mech):
    return model.with_missingness(mech)


def task_real(spec: dict) -> dict:
    """Classification of the real-gap series: oracles, or one unsupervised start."""
    subj, K, rule, kind, what = spec["subject"], spec["K"], spec["rule"], spec["model"], spec["what"]
    d = pc.windows(spec["data_dir"], subj)
    miss = d[pc.RULES[rule]].astype(bool)
    Y = d["Y_complete"].copy()
    Y[miss] = np.nan
    g = pc.groups(d, K)
    rows = []
    base = {"setting": "real", "subject": subj, "K": K, "rule": rule, "model": kind}
    if what == "oracle":
        om = oracle(spec["data_dir"], subj, K, kind)
        for mech in pc.MECHANISMS:
            est = pc.mechanism_from_labels(miss, g, K, mech)
            r, _ = _score(_with_mech(om, est), Y, g, K, miss, align=False,
                          base={**base, "fit": "oracle", "mechanism": mech, "start": 0},
                          labels=d["labels"] if spec.get("by_activity") else None)
            rows += r
        return {"classification": rows}
    mech, start = spec["mechanism"], spec["start"]
    init, tag = rc.starting_model(kind, K, Y, start, pc.N_STARTS)
    rows = []
    fitted = _unsupervised(init, Y, g, K, miss, kind, start, mech, "start",
                           {**base, "start": start}, rows)
    if mech == "ignorable":
        # the LR test's protocol: each mechanism estimated from the ignorable fit
        for mech2 in ("state", "state-markov"):
            _unsupervised(fitted, Y, g, K, miss, kind, start, mech2, "ignorable fit",
                          {**base, "start": start}, rows)
    return {"classification": rows}


def _unsupervised(init, Y, g, K, miss, kind, start, mech, start_from, base, rows,
                  truth=None, impute_seed=None, imp_rows=None):
    """One unsupervised ICE fit (settings of the real-series study) and its scores."""
    t0 = time.perf_counter()
    fitted, trace = ice(init, Y, {**_ice_cfg(kind, start), "max_iter": pc.ICE_MAX_ITER,
                                  "tol": pc.ICE_TOL, "missingness": mech})
    fit_s = time.perf_counter() - t0
    means = pc.state_means(fitted)
    r, b = _score(fitted, Y, g, K, miss, align=True,
                  base={**base, "fit": "unsupervised", "mechanism": mech, "start_from": start_from,
                        "n_iter": len(trace.log_liks), "fit_s": fit_s,
                        "state_means": "|".join(f"{v:.4f}" for v in np.sort(means))},
                  truth=truth, impute_seed=impute_seed)
    r[0].update({f"sorted_{k}": v for k, v in _sorted_params(fitted)[0].items()})
    r[0]["ll_mask_common"] = _mask_ll_common(miss, K)
    rows += r
    if imp_rows is not None:
        imp_rows += b
    return fitted


def _mask_ll_common(miss, K) -> str:
    """log p(m) of the common "state" and "state-markov" mechanisms (to compare
    an ignorable fit, log p(y_obs), with the MNAR fits' log p(y_obs, m))."""
    from pmcprg.pmc.missingness import common_mechanism, mask_log_likelihood
    return "|".join(f"{mask_log_likelihood(common_mechanism(m, miss, K), miss):.4f}"
                    for m in ("state", "state-markov"))


def controlled_mask(d: dict, K: int, intensity: str, mask_mech: str, r: int, subj: int):
    """Mask drawn on the true group path (transient = its own class K), with
    the plain per-class rates of the real mask of rule ``intensity``."""
    g = pc.groups(d, K)
    cls = np.where(g < 0, K, g)
    par = pc.class_rates(d[pc.RULES[intensity]].astype(bool), cls, K + 1)
    m = _draw(mask_mech, cls, par, pc.seed("mask", subj, K, intensity, mask_mech, r))
    return m, par


def task_controlled(spec: dict) -> dict:
    subj, K, intensity, mask_mech, r = (spec["subject"], spec["K"], spec["intensity"],
                                        spec["mask"], spec["r"])
    d = pc.windows(spec["data_dir"], subj)
    m, par = controlled_mask(d, K, intensity, mask_mech, r, subj)
    truth = d["Y_complete"]
    Y = truth.copy()
    Y[m] = np.nan
    g = pc.groups(d, K)
    g3 = pc.groups(d, 3)
    base0 = {"setting": "controlled", "subject": subj, "K": K, "rule": intensity,
             "mask": mask_mech, "r": r, "M": int(m.sum()),
             "mask_rates": "|".join(f"{v:.4g}" for v in par["rates"]),
             "mask_onset": "|".join(f"{v:.4g}" for v in par["onset"]),
             "mask_persistence": "|".join(f"{v:.4g}" for v in par["persistence"])}
    cls_rows, imp_rows = [], []
    do_imp = spec.get("impute", True)
    for kind in spec["oracles"]:
        om = oracle(spec["data_dir"], subj, K, kind)
        for mech in pc.MECHANISMS:
            est = pc.mechanism_from_labels(m, g, K, mech)
            a, b = _score(_with_mech(om, est), Y, g, K, m, align=False,
                          base={**base0, "model": kind, "fit": "oracle", "mechanism": mech,
                                "_g3": g3},
                          truth=truth if do_imp else None,
                          impute_seed=pc.seed("ffbs", subj, K, intensity, mask_mech, r, kind, "oracle", mech))
            cls_rows += a
            imp_rows += b
    kind = "hmc_in"
    init, _ = rc.starting_model(kind, K, Y, 0, 1)
    base = {**base0, "model": kind, "start": 0, "_g3": g3}
    ign = None
    for mech in pc.MECHANISMS:
        f = _unsupervised(init, Y, g, K, m, kind, 0, mech, "start", base, cls_rows,
                          truth=truth if do_imp else None,
                          impute_seed=pc.seed("ffbs", subj, K, intensity, mask_mech, r, kind,
                                              "unsup", mech),
                          imp_rows=imp_rows)
        ign = f if mech == "ignorable" else ign
    for mech in ("state", "state-markov"):
        _unsupervised(ign, Y, g, K, m, kind, 0, mech, "ignorable fit", base, cls_rows,
                      truth=truth if do_imp else None,
                      impute_seed=pc.seed("ffbs", subj, K, intensity, mask_mech, r, kind,
                                          "unsup-warm", mech),
                      imp_rows=imp_rows)
    return {"classification": cls_rows, "imputation": imp_rows}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TASKS = {"describe": task_describe, "test": task_test, "real": task_real,
         "controlled": task_controlled}


def run_task(spec: dict):
    logging.getLogger("pmcprg").setLevel(logging.ERROR)
    t0, c0 = time.perf_counter(), time.process_time()
    try:
        res = TASKS[spec["task"]](spec)
    except Exception as exc:  # recorded; the run goes on
        res = {"errors": [{"task": spec["task"],
                           "spec": json.dumps({k: v for k, v in spec.items() if k != "data_dir"}),
                           "error": f"{type(exc).__name__}: {exc}"[:500],
                           "traceback": traceback.format_exc()[-2000:]}]}
    return spec, res, (time.perf_counter() - t0, time.process_time() - c0)


def plan(args) -> list[dict]:
    subjects = (105,) if args.quick else pc.SUBJECTS
    parts = args.part
    specs = []
    if "describe" in parts:
        specs += [{"task": "describe", "subject": s} for s in subjects]
    if "test" in parts:
        for s in subjects:
            for K in pc.KS:
                for rule in pc.RULES:
                    B = args.boot if rule == "ge10" else args.boot_any
                    if args.boot_subjects and s not in args.boot_subjects:
                        B = 0
                    for alt, null in TESTS:
                        specs.append({"task": "test", "subject": s, "K": K, "rule": rule,
                                      "model": "hmc_in", "alternative": alt, "null": null,
                                      "B": 2 if args.quick else B,
                                      "statistic": STATISTIC})
                if args.test_pmc and K == 2:
                    for alt, null in TESTS:
                        specs.append({"task": "test", "subject": s, "K": K, "rule": "ge10",
                                      "model": "pmc_state", "alternative": alt, "null": null,
                                      "B": 0})
    if "real" in parts:
        for s in subjects:
            for K in pc.KS:
                for rule in pc.RULES:
                    for kind in ORACLE_KINDS:
                        specs.append({"task": "real", "what": "oracle", "by_activity": True,
                                      "subject": s, "K": K,
                                      "rule": rule, "model": kind})
                    for mech in pc.MECHANISMS:
                        for start in range(1 if args.quick else pc.N_STARTS):
                            specs.append({"task": "real", "what": "fit", "subject": s, "K": K,
                                          "rule": rule, "model": "hmc_in", "mechanism": mech,
                                          "start": start})
                    if K == 3 and not args.quick:
                        # copula model: k-means start, ignorable fit + the two
                        # mechanisms estimated from it
                        specs.append({"task": "real", "what": "fit", "subject": s, "K": K,
                                      "rule": rule, "model": "pmc_pair", "mechanism": "ignorable",
                                      "start": 0})
    if "controlled" in parts:
        R = 1 if args.quick else args.seeds
        for s in subjects:
            for K in pc.KS:
                for intensity in pc.RULES:
                    for mask in ("state", "state-markov"):
                        for r in range(R):
                            specs.append({"task": "controlled", "subject": s, "K": K,
                                          "intensity": intensity, "mask": mask, "r": r,
                                          "oracles": list(ORACLE_KINDS) if K == 3 else ["hmc_in"]})
    for sp in specs:
        sp["data_dir"] = args.data
        sp["quick"] = args.quick
    # longest first
    def cost(sp):
        if sp["task"] == "test":
            return 3 * (sp["B"] + 3)
        if sp["task"] == "real":
            return 60 if sp["model"] == "pmc_pair" and sp["what"] == "fit" else 2
        return {"controlled": 15, "describe": 40}[sp["task"]]
    specs.sort(key=lambda sp: -cost(sp))
    return specs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data", default=pc.DEFAULT_DATA)
    ap.add_argument("--part", nargs="+", default=["describe", "test", "real", "controlled"],
                    choices=["describe", "test", "real", "controlled"])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--boot", type=int, default=99, help="bootstrap B, rule ge10")
    ap.add_argument("--boot-any", type=int, default=0, help="bootstrap B, rule any")
    ap.add_argument("--boot-subjects", type=int, nargs="*", default=None,
                    help="bootstrap only these subjects (others: B = 0)")
    ap.add_argument("--seeds", type=int, default=CONTROLLED_SEEDS)
    ap.add_argument("--test-pmc", action="store_true")
    ap.add_argument("--quick", action="store_true", help="smoke run: subject 105, tiny B")
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--cache", type=Path, default=pc.REPO / "report" / "out" / "missing_state_pamap2",
                    help="task cache (git-ignored); a rerun skips cached tasks")
    ap.add_argument("--force", action="store_true", help="recompute cached tasks")
    args = ap.parse_args()
    out = args.out if not args.quick else args.out.parent / "results_quick"
    out.mkdir(parents=True, exist_ok=True)
    cache = args.cache / ("quick" if args.quick else "full")
    cache.mkdir(parents=True, exist_ok=True)
    specs = plan(args)
    collected: dict[str, list] = {}
    timing = []

    def spec_key(sp):
        body = json.dumps({k: v for k, v in sp.items() if k != "data_dir"}, sort_keys=True)
        return cache / f"{sp['task']}_{hashlib.sha1(body.encode()).hexdigest()[:16]}.pkl"

    def collect(spec, res, dt, cpu, cached):
        for k, rows in res.items():
            collected.setdefault(k, []).extend(rows)
        timing.append({"task": spec["task"], "seconds": dt, "cpu_seconds": cpu, "cached": cached,
                       "spec": json.dumps({k: v for k, v in spec.items() if k != "data_dir"})})

    todo = []
    for sp in specs:
        p = spec_key(sp)
        if p.exists() and not args.force:
            spec, res, (dt, cpu) = pickle.loads(p.read_bytes())
            collect(spec, res, dt, cpu, True)
        else:
            todo.append(sp)
    print(f"{len(specs)} tasks ({len(specs) - len(todo)} cached), {args.jobs} processes", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(run_task, sp) for sp in todo]
        for i, f in enumerate(as_completed(futs), 1):
            spec, res, (dt, cpu) = f.result()
            collect(spec, res, dt, cpu, False)
            if "errors" in res:
                print("ERROR", res["errors"][0]["error"], flush=True)
            else:
                spec_key(spec).write_bytes(pickle.dumps((spec, res, (dt, cpu))))
            if i % 10 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} done, {time.perf_counter() - t0:.0f} s", flush=True)
    names = {"desc_100hz": "describe_100hz", "desc_windows": "describe_windows", "gof": "gof",
             "sup_mech": "supervised_mechanism", "sup_lr": "supervised_lr", "tests": "lr_tests",
             "classification": "classification", "imputation": "imputation", "errors": "errors"}
    for k, rows in collected.items():
        df = pd.DataFrame(rows)
        if k == "imputation":       # the mask and fit details are in classification.csv
            df = df[[c for c in IMPUTATION_COLUMNS if c in df]]
        path = out / f"{names[k]}.csv"
        if k in ("classification", "imputation") and path.exists() and set(args.part) != {
                "describe", "test", "real", "controlled"}:
            old = pd.read_csv(path)
            keep_old = ~old["setting"].isin(df["setting"].unique())
            df = pd.concat([old[keep_old], df], ignore_index=True)
        df.to_csv(path, index=False, float_format="%.6g")
        print(f"wrote {path} ({len(df)} rows)")
    tpath = out / "tasks.csv"
    tdf = pd.DataFrame(timing)
    if tpath.exists():                # keep the rows of tasks outside this run
        old = pd.read_csv(tpath)
        if "cached" in old:
            old = old[~old["spec"].isin(tdf["spec"])]
            tdf = pd.concat([old, tdf], ignore_index=True)
    tdf.to_csv(tpath, index=False, float_format="%.6g")
    if not todo:          # everything from the cache: keep the run_info of the computing run
        return
    info = {"parts": args.part, "jobs": args.jobs, "boot": args.boot, "boot_any": args.boot_any,
            "seeds": args.seeds, "wall_s": time.perf_counter() - t0,
            "tasks_computed": len(todo),
            "task_wall_s": float(sum(t["seconds"] for t in timing if not t["cached"])),
            "task_cpu_s": float(sum(t["cpu_seconds"] for t in timing if not t["cached"])),
            "python": platform.python_version(), "machine": platform.machine(),
            "numpy": np.__version__}
    full = set(args.part) == {"describe", "test", "real", "controlled"}
    name = "run_info.json" if full else f"run_info_{'_'.join(args.part)}.json"
    (out / name).write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
