"""
pm_common.py — data, mask statistics, mechanisms and scores of the PAMAP2
study of state-dependent missingness (P6).

The 2 Hz series, labels, groups and gap rules are those of the real-series
study (``report/real_series/rs_common.py``, imported here, not copied):
windows of 50 samples of the 100 Hz hand accelerometer, Y = log-SD of ‖a‖,
K = 2 groups (rest / active) and K = 3 (rest / locomotion / vigorous),
activity 0 (transient) kept in the series and excluded from every score; a
window is missing when ≥ 5 of its 50 samples are NaN (rule ``ge10``) or when
any is (rule ``any``). The data are read from outside the repository and
never written into it.
"""

from __future__ import annotations

import logging
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (REPO, REPO / "report" / "real_series"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import rs_common as rc  # noqa: E402

from pmcprg.missing import metrics  # noqa: E402
from pmcprg.pmc.missingness import (  # noqa: E402
    StateMarkovMissingness,
    StateMissingness,
    estimate_state,
    estimate_state_markov,
)

DEFAULT_DATA = rc.DEFAULT_DATA
SUBJECTS = (102, 108, 105)
#: gap rules of the real-series study: key -> field of ``rc.prep_pamap2``.
RULES = {"ge10": "real", "any": "real_any"}
KS = (2, 3)
MECHANISMS = ("ignorable", "state", "state-markov")
GROUP_NAMES = {2: rc.GROUP_NAMES_K2, 3: rc.GROUP_NAMES_K3}
ACTIVITIES = rc.PAMAP2_ACTIVITIES
#: ICE settings of the unsupervised classification fits: those of the
#: real-series study (``rc.estimate``), plus the ``missingness`` key.
ICE_MAX_ITER = rc.ICE_MAX_ITER
ICE_TOL = rc.ICE_TOL
COPULA_CANDIDATES = rc.COPULA_CANDIDATES
N_STARTS = 3
RELIABILITY_BINS = 10

logger = logging.getLogger("pamap2_missing_state")


def seed(*parts) -> int:
    """A fixed 32-bit seed from a named tuple (``zlib.crc32``, never ``hash``)."""
    return int(zlib.crc32(repr(tuple(parts)).encode("utf-8")))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def windows(data_dir: str, subject: int) -> dict:
    """The 2 Hz data of the real-series study (``rc.prep_pamap2``, full series)."""
    return rc.prep_pamap2(data_dir, subject, False)


def groups(d: dict, K: int) -> np.ndarray:
    """Group of every window (−1: transient)."""
    return d["g2"] if K == 2 else d["g3"]


def load_100hz(data_dir: str, subject: int) -> tuple[np.ndarray, np.ndarray]:
    """(activity id, hand mask) of every 100 Hz row."""
    df = pd.read_csv(Path(data_dir) / f"pamap2/output/subject{subject}_100Hz.csv",
                     usecols=["X", "Y"])
    return df["X"].to_numpy(int), ~np.isfinite(df["Y"].to_numpy(float))


def group_of_activity(act: np.ndarray, K: int) -> np.ndarray:
    """Group (K = 2 or 3) of every activity id; −1 for transient (0)."""
    table = rc.GROUPS_K2 if K == 2 else rc.GROUPS_K3
    lut = np.full(int(act.max()) + 1, -1, dtype=int)
    for a, g in table.items():
        if a < lut.size:
            lut[a] = g
    return lut[act]


# ---------------------------------------------------------------------------
# Mask statistics
# ---------------------------------------------------------------------------

def bursts(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(start index, length) of every run of missing rows."""
    m = np.asarray(m, bool).astype(np.int8)
    dm = np.diff(np.concatenate([[0], m, [0]]))
    starts = np.nonzero(dm == 1)[0]
    ends = np.nonzero(dm == -1)[0]
    return starts, ends - starts


def transition_counts(m: np.ndarray, cls: np.ndarray, c) -> dict:
    """Mask transitions n−1 → n (n ≥ 1) whose destination row n is in class c."""
    m = np.asarray(m, bool)
    prev, cur, sel = m[:-1], m[1:], cls[1:] == c
    return {"n00": int(np.sum(sel & ~prev & ~cur)), "n01": int(np.sum(sel & ~prev & cur)),
            "n10": int(np.sum(sel & prev & ~cur)), "n11": int(np.sum(sel & prev & cur))}


def _rate(k, n) -> float:
    return float(k) / float(n) if n > 0 else float("nan")


def block_dispersion(m: np.ndarray, cls: np.ndarray, c, block: int) -> float:
    """Index of dispersion (variance / mean) of the missing counts over the
    consecutive blocks of ``block`` rows lying entirely in class c."""
    n = (len(m) // block) * block
    mb = np.asarray(m[:n], float).reshape(-1, block).sum(axis=1)
    cb = np.asarray(cls[:n]).reshape(-1, block)
    full = np.all(cb == c, axis=1)
    x = mb[full]
    if x.size < 2 or x.mean() == 0:
        return float("nan")
    return float(x.var(ddof=1) / x.mean())


def mask_stats(m: np.ndarray, cls: np.ndarray, c, *, hist_edges=(1, 2, 6, 21),
               block: int | None = None) -> dict:
    """Missing rate, bursts, onset / persistence of class c (a burst belongs to
    the class of its first row)."""
    m = np.asarray(m, bool)
    sel = cls == c
    starts, lengths = bursts(m)
    own = cls[starts] == c
    L = lengths[own]
    tc = transition_counts(m, cls, c)
    a = _rate(tc["n01"], tc["n00"] + tc["n01"])
    b = _rate(tc["n11"], tc["n10"] + tc["n11"])
    pi = _rate(int(np.sum(m & sel)), int(sel.sum()))
    edges = list(hist_edges) + [np.inf]
    hist = [int(np.sum((L >= lo) & (L < hi))) for lo, hi in zip(edges[:-1], edges[1:])]
    out = {"n": int(sel.sum()), "n_missing": int(np.sum(m & sel)), "rate": pi,
           "n_bursts": int(L.size),
           "burst_mean": float(L.mean()) if L.size else float("nan"),
           "burst_median": float(np.median(L)) if L.size else float("nan"),
           "burst_max": int(L.max()) if L.size else 0,
           "burst_hist": "|".join(map(str, hist)),
           "share_len1": float(np.mean(L == 1)) if L.size else float("nan"),
           "onset": a, "persistence": b,
           "geom_mean_len": 1.0 / (1.0 - b) if np.isfinite(b) and b < 1 else float("nan"),
           "b_over_pi": b / pi if pi and np.isfinite(b) else float("nan"), **tc}
    if block:
        out["dispersion"] = block_dispersion(m, cls, c, block)
    return out


# ---------------------------------------------------------------------------
# Mechanisms from the labels (supervised view)
# ---------------------------------------------------------------------------

def one_hot(g: np.ndarray, K: int) -> np.ndarray:
    """(N, K) one-hot posteriors of the groups; transient rows (−1) are zero."""
    G = np.zeros((g.size, K))
    keep = g >= 0
    G[np.nonzero(keep)[0], g[keep]] = 1.0
    return G


def mechanism_from_labels(miss: np.ndarray, g: np.ndarray, K: int, mechanism: str,
                          pseudo_count: float | None = None):
    """The library's M-step of ``mechanism`` on the one-hot label posteriors
    (transient rows carry no weight) — the supervised estimate."""
    if mechanism == "ignorable":
        return None
    G = one_hot(g, K)
    if mechanism == "state":
        return estimate_state(G, miss, pseudo_count=pseudo_count)
    return estimate_state_markov(G, miss, pseudo_count=pseudo_count)


def class_rates(miss: np.ndarray, cls: np.ndarray, n_classes: int) -> dict:
    """Plain (unguarded) π_c, a_c, b_c of every class 0..n_classes−1 — the
    parameters of the simulated masks (a class without a burst gets b = 0)."""
    pi, a, b = [], [], []
    for c in range(n_classes):
        sel = cls == c
        tc = transition_counts(miss, cls, c)
        pi.append(_rate(int(np.sum(miss & sel)), int(sel.sum())))
        a.append(_rate(tc["n01"], tc["n00"] + tc["n01"]))
        bb = _rate(tc["n11"], tc["n10"] + tc["n11"])
        b.append(0.0 if not np.isfinite(bb) else bb)
    pi, a = np.nan_to_num(pi), np.nan_to_num(a)
    return {"rates": pi, "onset": a, "persistence": np.asarray(b)}


def supervised_lr(miss: np.ndarray, g: np.ndarray, K: int) -> list[dict]:
    """Likelihood-ratio tests of the mask given the TRUE groups (non-transient
    rows; plain MLE, no guard). "state" vs common: Bernoulli rates;
    "state-markov" terms: transitions n−1 → n with row n non-transient
    (conditional on m_0); the nested test compares the Markov and Bernoulli
    likelihoods of the same rows n ≥ 1. The Bernoulli tests treat the rows
    as independent given the groups — anticonservative on bursty masks."""
    from scipy.stats import chi2

    def xlogy(k, p):
        return k * np.log(p) if k > 0 else 0.0

    def bern(k, n):
        p = k / n if n else 0.0
        return xlogy(k, p) + xlogy(n - k, 1 - p)

    # Bernoulli on all kept rows
    Mk = [int(np.sum(miss & (g == k))) for k in range(K)]
    Nk = [int(np.sum(g == k)) for k in range(K)]
    ll1 = sum(bern(Mk[k], Nk[k]) for k in range(K))
    ll0 = bern(sum(Mk), sum(Nk))
    rows = [{"alternative": "state", "null": "common", "LR": 2 * (ll1 - ll0), "df": K - 1}]
    # rows n ≥ 1, destination kept
    tcs = [transition_counts(miss, g, k) for k in range(K)]

    def markov(tc):
        return bern(tc["n01"], tc["n00"] + tc["n01"]) + bern(tc["n11"], tc["n10"] + tc["n11"])

    pooled = {key: sum(tc[key] for tc in tcs) for key in ("n00", "n01", "n10", "n11")}
    llm1 = sum(markov(tc) for tc in tcs)
    llm0 = markov(pooled)
    rows.append({"alternative": "state-markov", "null": "common", "LR": 2 * (llm1 - llm0),
                 "df": 2 * (K - 1)})
    lls = sum(bern(tc["n01"] + tc["n11"], sum(tc.values())) for tc in tcs)
    rows.append({"alternative": "state-markov", "null": "state", "LR": 2 * (llm1 - lls),
                 "df": K})
    for r in rows:
        r["p_chi2"] = float(chi2.sf(r["LR"], r["df"]))
    return rows


def mechanism_params(mech) -> dict:
    """Flat, '|'-joined parameters of a mechanism object (or ignorable)."""
    if mech is None:
        return {"rates": "", "onset": "", "persistence": ""}
    if isinstance(mech, StateMissingness):
        return {"rates": "|".join(f"{v:.5g}" for v in mech.rates), "onset": "", "persistence": ""}
    assert isinstance(mech, StateMarkovMissingness)
    return {"rates": "|".join(f"{v:.5g}" for v in mech.stationary),
            "onset": "|".join(f"{v:.5g}" for v in mech.onset),
            "persistence": "|".join(f"{v:.5g}" for v in mech.persistence)}


def permute_mechanism(mech, order):
    """The mechanism with its states reordered (``order[k]`` = old index of new k)."""
    if mech is None:
        return None
    if isinstance(mech, StateMissingness):
        return StateMissingness(rates=tuple(mech.rates[i] for i in order))
    return StateMarkovMissingness(onset=tuple(mech.onset[i] for i in order),
                                  persistence=tuple(mech.persistence[i] for i in order))


def state_means(model) -> np.ndarray:
    """Margin mean of every state (diagonal pair margin for pair models)."""
    K = model.K
    if model.margin_structure == "state":
        return np.array([model.margin(i)._frozen.mean() for i in range(K)])
    return np.array([model.margin(i, i)._frozen.mean() for i in range(K)])


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

def hungarian(x_true: np.ndarray, x_hat: np.ndarray, K: int) -> np.ndarray:
    """relabel such that relabel[x_hat] best matches x_true."""
    from scipy.optimize import linear_sum_assignment
    C = np.zeros((K, K), dtype=int)
    np.add.at(C, (x_true, x_hat), 1)
    rows, cols = linear_sum_assignment(-C)
    relabel = np.arange(K)
    relabel[cols] = rows
    return relabel


def classification_scores(g: np.ndarray, gamma: np.ndarray, K: int, miss: np.ndarray,
                          *, align: bool) -> dict:
    """MPM errors (all kept rows, missing, observed) and calibration at the
    missing rows: accuracy, mean confidence (max posterior), Brier score,
    log-loss of the true group, ECE and the reliability bins.

    ``align``: relabel the states by the Hungarian permutation of the MPM on
    every kept row (unsupervised fits); oracles are scored as they are.
    """
    keep = g >= 0
    x_hat = gamma.argmax(axis=1)
    relabel = hungarian(g[keep], x_hat[keep], K) if align else np.arange(K)
    G = np.zeros_like(gamma)
    G[:, relabel] = gamma                     # column relabel[k] <- old column k
    x = relabel[x_hat]
    wrong = (x != g) & keep
    out = {"err_all": float(wrong[keep].mean()), "n_all": int(keep.sum()),
           "perm": "".join(map(str, relabel)), "perm_identity": bool(np.all(relabel == np.arange(K)))}
    for name, sel in (("missing", miss & keep), ("observed", ~miss & keep)):
        out[f"err_{name}"] = float(wrong[sel].mean()) if sel.any() else float("nan")
        out[f"n_{name}"] = int(sel.sum())
    sel = miss & keep
    if sel.any():
        Gm, gm = G[sel], g[sel]
        conf = Gm.max(axis=1)
        correct = Gm.argmax(axis=1) == gm
        onehot = np.eye(K)[gm]
        out["acc_missing"] = float(correct.mean())
        out["conf_missing"] = float(conf.mean())
        out["brier_missing"] = float(np.mean(np.sum((Gm - onehot) ** 2, axis=1)))
        out["logloss_missing"] = float(np.mean(-np.log(np.clip(Gm[np.arange(gm.size), gm], 1e-12, 1.0))))
        out["p_true_missing"] = float(np.mean(Gm[np.arange(gm.size), gm]))
        edges = np.linspace(1.0 / K, 1.0, RELIABILITY_BINS + 1)
        idx = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, RELIABILITY_BINS - 1)
        n_b = np.bincount(idx, minlength=RELIABILITY_BINS)
        c_b = np.bincount(idx, weights=conf, minlength=RELIABILITY_BINS)
        a_b = np.bincount(idx, weights=correct.astype(float), minlength=RELIABILITY_BINS)
        out["ece_missing"] = float(np.sum(np.abs(c_b - a_b)) / sel.sum())
        out["rel_n"] = "|".join(map(str, n_b.astype(int)))
        out["rel_conf"] = "|".join(f"{v:.6g}" for v in c_b)
        out["rel_correct"] = "|".join(f"{v:.6g}" for v in a_b)
        # share of the missing rows assigned to each group vs the truth
        out["share_true_missing"] = "|".join(f"{np.mean(gm == k):.4f}" for k in range(K))
        out["share_post_missing"] = "|".join(f"{Gm[:, k].mean():.4f}" for k in range(K))
    return out


def imputation_scores(truth: np.ndarray, imp, sel: np.ndarray) -> dict:
    """RMSE / MAE of the posterior mean, CRPS of the FFBS draws, 90 % coverage
    and width, on the positions ``sel`` of ``imp.index``."""
    t = truth[sel]
    mean = imp.mean[sel]
    lo, hi = imp.quantile_values[sel, 0], imp.quantile_values[sel, -1]
    return {"n": int(sel.sum()),
            "rmse": float(np.sqrt(np.mean((t - mean) ** 2))),
            "mae": float(np.mean(np.abs(t - mean))),
            "crps": float(metrics.crps_from_samples(t, imp.y_samples.T[sel])),
            "cov90": float(metrics.interval_coverage(t, lo, hi)),
            "width90": float(np.mean(hi - lo))}
