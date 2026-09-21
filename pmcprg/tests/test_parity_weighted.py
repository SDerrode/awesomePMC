"""Weighted estimation and family selection against pyvinecopulib and VineCopula — audit FR-11, wave 3.

The copula fit of ICE's M-step (``pmcprg.pmc.ice``) weights each pseudo-
observation of the pair (i, j) by its posterior ξ_n(i, j): a weighted
Kendall τ (:func:`~pmcprg.pmc.ice._weighted_kendall_tau`, the start value and
fallback), a weighted MLE (:func:`~pmcprg.pmc.ice._fit_copula_params`) and a
weighted selection criterion (:func:`~pmcprg.pmc.ice._select_and_fit_copula`).
They are tested here as ICE calls them and, since FR-12 made them public,
through ``CopulaVirt.fit(..., weights=…)`` and ``fit_best`` — the same engine
(the last section below) — against

* pyvinecopulib 1.0.0 (``Bicop.fit`` with ``FitControlsBicop(weights=…)``,
  ``pyvinecopulib.utils.wdm``);
* VineCopula 2.6.1 (``BiCopEst(weights=…)``, ``TauMatrix``, ``BiCopSelect``);
* exact integer arithmetic (weighted τ) and a polished optimum of pmcprg's own
  weighted log-likelihood (MLE), written here.

Data and tables are generated offline (``scripts/parity/gen_weighted_data.py``,
``gen_pyvinecopulib_weighted.py``, ``gen_r_weighted.R``;
``scripts/parity/README.md``); nothing here needs R, pyvinecopulib or mpmath.
Every coordinate and weight of the data files is an integer (u = rank/(n + 1),
w = k/2²⁰), so R, Python and this module read the same doubles.

Definitions compared
--------------------
* **Weighted Kendall τ.** pyvinecopulib (``wdm``) and VineCopula
  (``TauMatrix``, which ``BiCopEst``'s ``emptau`` and itau use) compute
  Σ_{i<j} wᵢwⱼ sgn(Δu) sgn(Δv) over √(Σ wᵢwⱼ sgn(Δu)² · Σ wᵢwⱼ sgn(Δv)²) — a
  weighted τ-b. pmcprg computed the weighted τ-a (the same numerator over
  Σ_{i<j} wᵢwⱼ) until FR-12, and computes the τ-b since: **without ties the
  two are the same number** (measured below; the τ-a code is kept for that
  case, bit for bit), with ties τ-b is 4.6·10⁻² to 5.1·10⁻² above τ-a on an
  8 × 8 grid (:func:`test_references_weight_tau_b_on_ties`). ICE's
  pseudo-observations come from continuous margins: no ties.
* **Weighted log-likelihood.** pmcprg (``_weighted_log_likelihood``, the
  selection scores and the joint MLE): the **total** Σ wᵢ log c(uᵢ, vᵢ);
  the one-parameter search minimises the same sum divided by Σw (same
  optimum). VineCopula's MLE: the same total (its ``logLik``), but its
  **itau** ``logLik`` ignores the weights (Σ log c over every row, zero
  weights included). pyvinecopulib rescales the weights to mean 1 **over the
  rows, zero weights included**: its ``loglik()`` is (n_rows/Σw)·Σ wᵢ log cᵢ,
  while its ``nobs`` counts the positive weights — so a {0, 1}-weighted fit
  has the subset's parameters but not the subset's log-likelihood
  (:func:`test_reference_log_likelihood_definitions`).
* **Selection penalties.** pmcprg: AIC 2k − 2ℓ, BIC k·log(Σw) − 2ℓ (the
  effective size of a ξ-weighted block); VineCopula's ``BiCopSelect``: the
  same AIC, BIC with k·log(n_rows). Parameter counts agree (Product 0,
  one-parameter families 1, Student and BB1 2). The only selection
  disagreement measured is this BIC sample size (below).

Measured agreement (the tolerances are small multiples of these)
----------------------------------------------------------------
Seven samples of n = 300 (Gaussian, Student, Clayton and its 90° rotation,
Gumbel, Frank, BB1; |τ| ≈ 0.41–0.49) × four weightings (unit; generic
Beta(½, ½) weights, Σw ≈ 138–161; {0, 1} weights; the subset they keep),
plus the Gaussian sample on an 8 × 8 grid (ties):

* weighted τ against the exact value — pmcprg 1.7·10⁻¹⁶ (1.6·10⁻¹⁵ with the
  weights scaled), pyvinecopulib 5.6·10⁻¹⁷, VineCopula 1.8·10⁻¹³ (its R matrix
  product);
* a reference's itau parameter taken back to τ — 3.0·10⁻¹⁵ for the closed
  forms; Frank 1.1·10⁻¹⁰ (pyvinecopulib's numerical θ(τ)), and VineCopula's
  θ(τ) is its linearly interpolated table: 7.1·10⁻⁴ in τ through the exact
  τ(θ), 1.5·10⁻¹⁰ through its own table (:data:`ITAU_LIMITED`);
* weighted MLE, how far below the polished maximum of pmcprg's own weighted
  log-likelihood each optimum lies — pmcprg 1.3·10⁻¹¹ nat, pyvinecopulib
  2.6·10⁻¹⁰, VineCopula 3.0·10⁻⁸ (R's ``optimize`` at tol = 1.2·10⁻⁴ in the
  parameter). pmcprg is never below a reference by more than its own
  1.3·10⁻¹¹: **no implementation fails to maximise**, VineCopula stops the
  earliest;
* {0, 1} weights ≡ the subset: bit-identical in pmcprg (τ and every
  parameter); the references agree with their own subset fits to their
  shortfalls;
* weight scaling: invariant at every scale tried, Σw down to 1.3·10⁻⁴ —
  one-parameter fits τ̂ 5.5·10⁻¹², ℓ 2.4·10⁻¹³; two-parameter fits ℓ
  9.5·10⁻¹², τ̂ 1.4·10⁻⁷ since the joint MLE's ``gtol`` is scaled by the mean
  positive weight (found here: with a fixed 1e-6 a fit at Σw = 1.4·10⁻⁴
  stopped 2.6·10⁻³ nat short of its maximum).

Selection: eight families at |τ| ≈ 0.4 (Gauss, Student ν = 4, Clayton,
Clayton 90°, Gumbel, Frank, Joe, BB1), n = 500, twenty seeds, unweighted and
with deterministic weights, candidates {Product, Gauss, true family}, 960
decisions under ``mle``, ``aic`` and ``bic``. pmcprg recovers the true family
every time but for BB1 (unweighted BIC 19/20; weighted AIC 19/20, BIC 15/20 —
taken for Gauss); VineCopula's ``BiCopSelect`` makes the same choice in 959
of the 960, the other being the BIC sample size above. Each candidate's fit:
VineCopula up to 3.5·10⁻⁷ nat below pmcprg, pmcprg never more than
1.5·10⁻¹² below VineCopula. The smallest winning margin is 0.02 nat, so the
decisions do not hang on rounding (same under the minimum versions).

Discrepancies, each explained
-----------------------------
* τ-a against τ-b on ties — definitions (above); resolved by FR-12, pmcprg
  computes the references' τ-b.
* The log-likelihood each package reports — definitions (above).
* BIC's sample size — definitions; the one differing decision.
* References' precision: VineCopula's one-parameter MLE (3.0·10⁻⁸ nat short),
  pyvinecopulib's (2.6·10⁻¹⁰), VineCopula's Frank itau table (7.1·10⁻⁴ in τ),
  pyvinecopulib's Frank θ(τ) (1.1·10⁻¹⁰), VineCopula's τ sum (1.8·10⁻¹³);
  VineCopula's Student itau searches ν in [2, 10] only, at ``tol = 1``.
* pmcprg: the joint (two-parameter) MLE's stopping rule was not scale-free —
  L-BFGS-B's absolute ``gtol = 1e-6`` on the total objective stops early
  when Σw is small (loss up to 3.9·10⁻⁹ nat of the unscaled likelihood at
  Σw ≈ 1.4·10⁻², up to 2.6·10⁻³ with τ̂ off by up to 5.6·10⁻⁴ at
  Σw ≈ 1.4·10⁻⁴). Of no statistical consequence at those Σw, but the
  estimate was not invariant. Fixed at the author's decision: ``gtol`` is
  scaled by the mean positive weight, unit weights unchanged bit for bit.
"""
from __future__ import annotations

import csv
import functools
import hashlib
import json
import math
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest
import scipy
from scipy.optimize import minimize, minimize_scalar

from pmcprg.copulas import CopulaEnum, CopulaVirt
from pmcprg.copulas.archimedean.frank import kendall_tau_frank
from pmcprg.copulas.archimedean.joe import _joe_tau_from_theta
from pmcprg.pmc.ice import (
    FIT_FAILED_KEY,
    _copula_candidate_fits,
    _fit_copula_params,
    _score_aic,
    _score_bic,
    _select_and_fit_copula,
    _weighted_kendall_tau,
    _weighted_log_likelihood,
)

DATA = Path(__file__).parent / "data" / "parity"
W_SCALE = 2 ** 20
SCHEMES = ("unit", "gen", "w01", "subset")
REFERENCES = ("pyvinecopulib", "VineCopula")

# pmcprg's registry entry of each data set's family.
ENTRY = {
    "gaussian": CopulaEnum.GAUSSIAN, "student": CopulaEnum.STUDENT,
    "clayton": CopulaEnum.CLAYTON, "clayton90": CopulaEnum.CLAYTON90,
    "gumbel": CopulaEnum.GH, "frank": CopulaEnum.FRANK, "joe": CopulaEnum.JOE,
    "bb1": CopulaEnum.BB1,
}
SHORT = {k: e.value.SHORT_NAME for k, e in ENTRY.items()}
# VineCopula family codes (README; 23 is Clayton's 90° rotation, parameter −θ).
VC_CODE = {"Prod": 0, "Gauss": 1, "Student": 2, "Clayton": 3, "Clayton90": 23, "GH": 4,
           "Frank": 5, "Joe": 6, "BB1": 7}


# ---------------------------------------------------------------------------
# Data and tables
# ---------------------------------------------------------------------------

def _csv_rows(name: str) -> list[dict]:
    with open(DATA / name, newline="") as fh:
        return list(csv.DictReader(line for line in fh if not line.startswith("#")))


def _ints(s: str) -> np.ndarray:
    return np.array([int(x) for x in s.split()], dtype=np.int64)


@functools.lru_cache(maxsize=None)
def _weighted_data() -> dict:
    out = {}
    for r in _csv_rows("weighted_data.csv"):
        den = int(r["denom"])
        ui, vi = _ints(r["u_int"]), _ints(r["v_int"])
        out[r["dataset"]] = {
            "family": r["family"], "rotation": int(r["rotation"]),
            "u": ui / den, "v": vi / den, "u_int": ui, "v_int": vi,
            "k_gen": _ints(r["w_gen"]), "k01": _ints(r["w01"]),
        }
    return out


def _scheme(dataset: str, scheme: str):
    """``(u, v, w, k)``: the sample, its float weights and the same weights as integers ``k``
    with ``w = k·(scale)`` — ``k`` feeds the exact weighted τ."""
    d = _weighted_data()[dataset]
    u, v = d["u"], d["v"]
    if scheme == "unit":
        k = np.ones(u.size, dtype=np.int64)
        return u, v, k.astype(float), k
    if scheme == "gen":
        return u, v, d["k_gen"] / W_SCALE, d["k_gen"]
    if scheme == "w01":
        return u, v, d["k01"].astype(float), d["k01"]
    if scheme == "subset":
        keep = d["k01"] > 0
        k = np.ones(int(keep.sum()), dtype=np.int64)
        return u[keep], v[keep], k.astype(float), k
    raise KeyError(scheme)


@functools.lru_cache(maxsize=None)
def _selection_rows() -> tuple:
    return tuple(_csv_rows("selection_data.csv"))


def _selection_sample(r: dict, weighting: str):
    n = int(r["n"])
    u = np.arange(1, n + 1) / (n + 1)
    v = _ints(r["v_by_u"]) / (n + 1)
    if weighting == "unit":
        w = np.ones(n)
    else:   # the deterministic weights of gen_r_weighted.R, by u-rank i
        w = ((7919 * np.arange(1, n + 1)) % 1024 + 1) / 1024
    return u, v, w


@functools.lru_cache(maxsize=None)
def _table(name: str) -> dict:
    with open(DATA / name) as fh:
        return json.load(fh)


def _pv() -> dict:
    return _table("weighted_pyvinecopulib.json")


def _vc() -> dict:
    return _table("weighted_vinecopula.json")


def _ref_fit(reference: str, dataset: str, scheme: str, method: str) -> dict | None:
    tab = _pv() if reference == "pyvinecopulib" else _vc()
    hits = [f for f in tab["fits"]
            if (f["dataset"], f["scheme"], f["method"]) == (dataset, scheme, method)]
    assert len(hits) <= 1
    return hits[0] if hits else None


def _ref_kendall(reference: str, dataset: str, scheme: str) -> float:
    if reference == "pyvinecopulib":
        return _pv()["kendall"][dataset][scheme]
    return _vc()["kendall"][dataset][scheme]["fasttau"]


def _params_of_key(short: str, pars: list) -> dict:
    """pmcprg's constructor parameters of the key's (README) parameters, by pmcprg's own τ maps."""
    if short == "Prod":
        return {"tau_k": 0.0}
    if short == "Gauss":
        return {"tau_k": 2.0 / math.pi * math.asin(pars[0])}
    if short == "Student":
        return {"tau_k": 2.0 / math.pi * math.asin(pars[0]), "df": pars[1]}
    if short == "Clayton":
        return {"tau_k": pars[0] / (pars[0] + 2.0)}
    if short == "Clayton90":
        return {"tau_k": -pars[0] / (pars[0] + 2.0)}
    if short == "GH":
        return {"tau_k": 1.0 - 1.0 / pars[0]}
    if short == "Frank":
        return {"tau_k": kendall_tau_frank(pars[0])}
    if short == "Joe":
        return {"tau_k": _joe_tau_from_theta(pars[0])}
    if short == "BB1":
        th, de = pars
        # 1 − 2/(δ(θ + 2)) without its cancellation (``_two_parameter_spec``).
        return {"tau_k": (de * th + 2.0 * (de - 1.0)) / (de * (th + 2.0)), "delta": de}
    raise KeyError(short)


def _vc_params(short: str, par: float, par2: float) -> dict:
    """The key of a VineCopula (code, par, par2) — its 90° codes take −θ — as pmcprg parameters."""
    if short == "Clayton90":
        return _params_of_key(short, [-par])
    return _params_of_key(short, [par] if short not in ("Student", "BB1") else [par, par2])


# ---------------------------------------------------------------------------
# The two oracles written here
# ---------------------------------------------------------------------------

def _exact_tau(u, v, k, tau_b: bool = False) -> float:
    """The weighted Kendall τ-a (τ-b) of integer weights ``k``, in exact integer arithmetic.

    Scale-free, so the τ of the float weights k·s. Numerator and denominators
    are exact int64 sums (|k| ≤ 2²⁰: every product ≤ 2⁴⁰, 44 850 of them
    < 2⁵⁶), the quotient Python's correctly rounded int division; τ-b's
    square root in 50-digit decimals.
    """
    su = np.sign(u[:, None] - u[None, :]).astype(np.int64)
    sv = np.sign(v[:, None] - v[None, :]).astype(np.int64)
    kk = np.asarray(k, dtype=np.int64)
    K = kk[:, None] * kk[None, :]
    iu = np.triu_indices(kk.size, 1)
    num = int((K * su * sv)[iu].sum())
    if not tau_b:
        return num / int(K[iu].sum())
    dx, dy = int((K * su * su)[iu].sum()), int((K * sv * sv)[iu].sum())
    with localcontext() as ctx:
        ctx.prec = 50
        return float(Decimal(num) / (Decimal(dx) * Decimal(dy)).sqrt())


def _optimum(entry, u, v, w, starts: list[dict]) -> float:
    """The maximum of pmcprg's weighted log-likelihood, polished from the best of ``starts``.

    One parameter: bounded Brent at ``xatol = 1e-12`` in τ within ±10⁻³ of
    the best start (the three implementations' τ̂ lie within 8.7·10⁻⁶ of each
    other). Two: Nelder–Mead in (τ, extra) at ``xatol = 1e-9`` and
    ``fatol = 1e-12`` — ℓ ≈ 30–110 carries ~10⁻¹⁴ of rounding, which a
    tighter ``fatol`` never meets (the search then runs to its iteration
    cap). The polish only ever raises the value.
    """
    cls = entry.klass

    def ll(p):
        return _weighted_log_likelihood(cls, p, u, v, w)

    best = max(starts, key=ll)
    extras = [k for k in best if k != "tau_k"]
    if not extras:
        t = best["tau_k"]
        res = minimize_scalar(lambda x: -ll({"tau_k": x}), bounds=(t - 1e-3, t + 1e-3),
                              method="bounded", options={"xatol": 1e-12})
        polished = {"tau_k": float(res.x)}
    else:
        (name,) = extras
        x0 = np.array([best["tau_k"], best[name]])
        simplex = [x0, x0 + [1e-3, 0.0], x0 + [0.0, 1e-2 * abs(x0[1])]]
        res = minimize(lambda x: -ll({"tau_k": x[0], name: x[1]}), x0, method="Nelder-Mead",
                       options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 4000,
                                "initial_simplex": simplex})
        polished = {"tau_k": float(res.x[0]), name: float(res.x[1])}
    return max(ll(best), ll(polished))


def _pmcprg_fit(dataset: str, u, v, w) -> dict:
    """pmcprg's weighted fit, as ICE's M-step calls it."""
    entry = ENTRY[dataset]
    params = _fit_copula_params(entry.klass, entry, u, v, w)
    assert FIT_FAILED_KEY not in params, (dataset, params)
    return params


# ---------------------------------------------------------------------------
# Tolerances — each a small multiple of the largest error measured on these
# data, under numpy 2.5.3 / scipy 1.18.1 and numpy 1.24.4 / scipy 1.10.1
# (the minimum versions; same figures unless stated).
# ---------------------------------------------------------------------------

# Weighted Kendall τ against the exact integer computation: pmcprg 1.7e-16
# (1.6e-15 with the weights scaled by 1/3), pyvinecopulib 5.6e-17 (both one
# ulp); VineCopula 1.8e-13 — ``TauMatrix`` sums the 44 850 pairs as an R
# matrix product of weights pre-divided by √(Σ w_i w_j).
TOL_TAU = {"pmcprg": 5e-15, "pyvinecopulib": 5e-16, "VineCopula": 5e-13}

# A reference's itau fit, taken back to τ — by its own τ(θ) or by pmcprg's
# exact map — against its own weighted τ: 3.0e-15 (the round trip of the
# closed-form maps). Frank has no closed-form θ(τ), and :data:`ITAU_LIMITED`.
TOL_ITAU = 1e-14
# (reference, how) → tolerance for Frank. pyvinecopulib inverts τ(θ)
# numerically (1.1e-10); VineCopula's ``BiCopTau2Par`` and ``BiCopPar2Tau``
# both go through the linearly interpolated table the pilot found (5.5e-4 on
# its grid): they undo each other to 1.5e-10 ("own"), but the θ is the
# table's — 7.1e-4 in τ by pmcprg's exact τ(θ) ("mapped").
ITAU_LIMITED = {
    ("pyvinecopulib", "own"): 3e-10, ("pyvinecopulib", "mapped"): 3e-10,
    ("VineCopula", "own"): 3e-10, ("VineCopula", "mapped"): 1e-3,
}

# Weighted MLE: how far below the polished maximum of pmcprg's weighted
# log-likelihood each implementation's optimum lies, in nats (Σw = 138 to 300):
# pmcprg 1.3e-11 (Brent at xatol 1e-6 in τ; the joint L-BFGS-B 5.1e-13),
# pyvinecopulib 2.6e-10 (Student), VineCopula 3.0e-8 — R's ``optimize`` at its
# default tol = 1.2e-4 in the parameter (Gaussian; Clayton 90° 5.9e-9).
SHORTFALL = {"pmcprg": 5e-11, "pyvinecopulib": 1e-9, "VineCopula": 1e-7}

# A log-likelihood a reference recorded against pmcprg's evaluation at the
# same parameters, relative: 5.5e-14 (the interior density parity of the
# pilot, summed). pmcprg's Student density rests on scipy's t.ppf, 2.3e-9
# relative off before scipy 1.13 (pilot, ``STUDENT_TOL_OLD_SCIPY``): 3.1e-10
# relative under scipy 1.10.1.
_SCIPY = tuple(int(x) for x in scipy.__version__.split(".")[:2])
TOL_LOGLIK_EVAL = 5e-13
TOL_LOGLIK_EVAL_STUDENT = 1e-9 if _SCIPY < (1, 17) else TOL_LOGLIK_EVAL

# Weight scaling, every c tried (1/3 … 1e6, down to 2^-20: Σw from 1.3e-4 to
# 1.4e8). One parameter: τ̂ moves 5.5e-12 at most (Brent sees the
# Σw-normalised objective, rounded differently), ℓ 2.4e-13. Two: since the
# joint MLE's gtol is scaled by the mean positive weight (FR-11 wave 3; with
# the fixed 1e-6 a fit at Σw = 1.4e-4 stopped 2.6e-3 nat short), ℓ 9.5e-12
# (Student) and 7.3e-12 (BB1), τ̂ 1.4e-7 along the flat direction of the
# likelihood (L-BFGS-B stops elsewhere on it).
TOL_SCALE = {"tau1": 2e-11, "loglik1": 1e-12, "tau2": 5e-7, "loglik2": 3e-11}

# Selection samples (n = 500): each candidate's pmcprg fit against
# VineCopula's, both evaluated by pmcprg. pmcprg is never lower by more than
# 1.5e-12; VineCopula is up to 3.5e-7 lower (the Gaussian's ``optimize``, as
# above, at n = 500).
TOL_SELECTION_LOGLIK = {"pmcprg": SHORTFALL["pmcprg"], "VineCopula": 1e-6}

WEIGHTED_SETS = ["gaussian", "student", "clayton", "clayton90", "gumbel", "frank", "bb1"]
SELECTION_SETS = ["gaussian", "student", "clayton", "clayton90", "gumbel", "frank", "joe", "bb1"]
FAST_SEEDS = range(1, 6)
SLOW_SEEDS = range(6, 21)
CRITERIA = {"mle": "logLik", "aic": "AIC", "bic": "BIC"}   # pmcprg → BiCopSelect

# pmcprg's failures to recover the true family (dataset, seed, weighting,
# criterion) → the family it picks; every other decision is the true family.
# All are BB1 (τ = 0.40, λ_L = 0.35, λ_U = 0.32) taken for Gauss once a
# penalty charges its second parameter: unweighted BIC 19/20; weighted AIC
# 19/20, BIC 15/20 (Σw = 242.6). VineCopula makes the same seven choices.
RECOVERY_FAILURES = {
    ("bb1", 1, "unit", "bic"): "Gauss",
    ("bb1", 1, "formula", "aic"): "Gauss",
    ("bb1", 1, "formula", "bic"): "Gauss",
    ("bb1", 3, "formula", "bic"): "Gauss",
    ("bb1", 5, "formula", "bic"): "Gauss",
    ("bb1", 8, "formula", "bic"): "Gauss",
    ("bb1", 10, "formula", "bic"): "Gauss",
}
# The one decision of 960 where VineCopula differs → its pick. Cause: BIC's
# sample size — pmcprg charges log Σw = log 242.6 per parameter, VineCopula
# log n = log 500 (:func:`test_selection_disagreement_is_the_bic_sample_size`).
SELECTION_DISAGREEMENTS = {("bb1", 12, "formula", "bic"): "Gauss"}


def _student_aware(dataset_or_short: str) -> float:
    return (TOL_LOGLIK_EVAL_STUDENT if dataset_or_short in ("student", "Student")
            else TOL_LOGLIK_EVAL)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def test_tables_carry_provenance_and_the_data_they_read():
    pv, vc = _pv(), _vc()
    assert pv["reference"]["package"] == "pyvinecopulib" and pv["reference"]["version"] == "1.0.0"
    assert vc["reference"]["package"] == "VineCopula" and vc["reference"]["version"] == "2.6.1"
    for tab in (pv, vc):
        for key in ("generated", "repository_version", "command", "environment", "definitions"):
            assert tab[key], key
    md5 = {name: hashlib.md5((DATA / name).read_bytes()).hexdigest()
           for name in ("weighted_data.csv", "selection_data.csv")}
    assert pv["data"]["md5"] == md5["weighted_data.csv"]
    assert vc["data"]["weighted"] == md5["weighted_data.csv"]
    assert vc["data"]["selection"] == md5["selection_data.csv"]
    # Every data set, weighting and method the tests read.
    assert set(pv["kendall"]) == set(vc["kendall"]) == set(_weighted_data())
    for ref in REFERENCES:
        for ds in WEIGHTED_SETS:
            for sc in SCHEMES:
                assert _ref_fit(ref, ds, sc, "mle") is not None, (ref, ds, sc)
                assert (_ref_fit(ref, ds, sc, "itau") is None) == (ds == "bb1"), (ref, ds, sc)
    keys = {(s["dataset"], int(s["seed"]), s["weighting"]) for s in vc["selection"]}
    assert keys == {(r["dataset"], int(r["seed"]), w) for r in _selection_rows()
                    for w in ("unit", "formula")}
    assert len(_selection_rows()) == len(SELECTION_SETS) * 20


# ---------------------------------------------------------------------------
# Weighted Kendall τ
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dataset", list(_weighted_data()))
def test_weighted_tau_is_the_product_weight_coefficient(dataset):
    """pmcprg's O(N log N) weighted τ equals the exact O(n²) coefficient, at every weighting and scale.

    τ-a on the samples without ties, τ-b on the tied grid (FR-12).
    """
    for scheme in SCHEMES:
        u, v, w, k = _scheme(dataset, scheme)
        exact = _exact_tau(u, v, k, tau_b=(dataset == "gaussian_ties"))
        for c in (1.0, 1 / 3, 7.0, 1e3, 1e6, 2.0 ** -20, 1e-6):
            got = _weighted_kendall_tau(u, v, w * c)
            assert abs(got - exact) <= TOL_TAU["pmcprg"], (scheme, c, got, exact)


@pytest.mark.parametrize("reference", REFERENCES)
@pytest.mark.parametrize("dataset", WEIGHTED_SETS)
def test_weighted_tau_parity(reference, dataset):
    """Without ties, pmcprg's τ-a, the references' τ-b and the exact value are one number."""
    for scheme in SCHEMES:
        u, v, w, k = _scheme(dataset, scheme)
        exact_a, exact_b = _exact_tau(u, v, k), _exact_tau(u, v, k, tau_b=True)
        assert exact_a == exact_b                      # no ties: the definitions coincide
        ref = _ref_kendall(reference, dataset, scheme)
        assert abs(ref - exact_b) <= TOL_TAU[reference], (scheme, ref, exact_b)
        ours = _weighted_kendall_tau(u, v, w)
        assert abs(ours - ref) <= TOL_TAU["pmcprg"] + TOL_TAU[reference], (scheme, ours, ref)
        if reference == "VineCopula":
            vc_tm = _vc()["kendall"][dataset][scheme]["TauMatrix"]
            if scheme in ("gen", "w01"):       # fasttau *is* TauMatrix with weights
                assert vc_tm == ref
            for method in ("itau", "mle"):     # BiCopEst's emptau is fasttau
                fit = _ref_fit(reference, dataset, scheme, method)
                if fit is not None:
                    assert fit["emptau"] == ref


def test_references_weight_tau_b_on_ties():
    """With ties pmcprg computes the references' weighted τ-b (FR-12; τ-a before).

    On the 8 × 8 grid (``gaussian_ties``) τ-b and τ-a differ by 4.6·10⁻² to
    5.1·10⁻², a definition. pmcprg's weighted τ is now the τ-b: measured
    5.6·10⁻¹⁷ from the exact value, 0 from pyvinecopulib, 1.3·10⁻¹⁴ from
    VineCopula (its own 1.8·10⁻¹³ error), and 6.1·10⁻¹⁶ with the weights
    scaled — the tolerances above.
    """
    for scheme in SCHEMES:
        u, v, w, k = _scheme("gaussian_ties", scheme)
        exact_a, exact_b = _exact_tau(u, v, k), _exact_tau(u, v, k, tau_b=True)
        assert 4e-2 < exact_b - exact_a < 6e-2, (scheme, exact_a, exact_b)
        ours = _weighted_kendall_tau(u, v, w)
        assert abs(ours - exact_b) <= TOL_TAU["pmcprg"]
        for reference in REFERENCES:
            ref = _ref_kendall(reference, "gaussian_ties", scheme)
            assert abs(ref - exact_b) <= TOL_TAU[reference]
            assert abs(ours - ref) <= TOL_TAU["pmcprg"] + TOL_TAU[reference], (scheme, reference)


@pytest.mark.parametrize("reference", REFERENCES)
@pytest.mark.parametrize("dataset", [d for d in WEIGHTED_SETS if d != "bb1"])
def test_itau_is_the_weighted_tau(reference, dataset):
    """A reference's itau parameter, taken back to τ, is its weighted τ — and so pmcprg's.

    pmcprg has no itau of its own: it is parametrised by τ, and the weighted τ
    *is* the itau estimate (ICE's start value and fallback). Student's ν from
    the references' itau (a profile likelihood; VineCopula's restricted to
    [2, 10] at ``tol = 1``) is not a τ and is not compared.
    """
    short = SHORT[dataset]
    for scheme in SCHEMES:
        fit = _ref_fit(reference, dataset, scheme, "itau")
        ref_tau = _ref_kendall(reference, dataset, scheme)
        u, v, w, _ = _scheme(dataset, scheme)
        ours = _weighted_kendall_tau(u, v, w)
        for how, value in (("own", fit["tau"]),
                           ("mapped", _params_of_key(short, fit["pars"])["tau_k"])):
            tol = ITAU_LIMITED[(reference, how)] if dataset == "frank" else TOL_ITAU
            assert abs(value - ref_tau) <= tol, (scheme, how, value, ref_tau)
            assert abs(value - ours) <= tol + TOL_TAU["pmcprg"] + TOL_TAU[reference], (scheme, how)


@pytest.mark.parametrize("reference", REFERENCES)
def test_itau_limited_entries_are_needed(reference):
    """Frank's itau register is still needed: the closed-form families meet TOL_ITAU, Frank does not."""
    worst = {"own": 0.0, "mapped": 0.0}
    for scheme in SCHEMES:
        fit = _ref_fit(reference, "frank", scheme, "itau")
        ref_tau = _ref_kendall(reference, "frank", scheme)
        worst["own"] = max(worst["own"], abs(fit["tau"] - ref_tau))
        worst["mapped"] = max(worst["mapped"],
                              abs(_params_of_key("Frank", fit["pars"])["tau_k"] - ref_tau))
    for how in ("own", "mapped"):
        assert TOL_ITAU < worst[how] <= ITAU_LIMITED[(reference, how)], (how, worst[how])


# ---------------------------------------------------------------------------
# Weighted MLE
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("dataset", WEIGHTED_SETS)
def test_weighted_mle_reaches_the_optimum(dataset, scheme):
    """Every optimum, evaluated by pmcprg's weighted log-likelihood, within its shortfall of the maximum.

    Log-likelihoods, not parameters: the optimisers' tolerances dominate the
    fitted values (the three τ̂ spread up to 8.6·10⁻⁶), not the likelihood.
    pmcprg's fit is the one ICE's M-step makes; the maximum is the polished
    optimum of :func:`_optimum`. pmcprg is also never below a reference by
    more than its own shortfall: where they differ, the reference stopped
    short (module docstring).
    """
    entry, short = ENTRY[dataset], SHORT[dataset]
    u, v, w, _ = _scheme(dataset, scheme)
    optima = {"pmcprg": _pmcprg_fit(dataset, u, v, w)}
    for reference in REFERENCES:
        optima[reference] = _params_of_key(short, _ref_fit(reference, dataset, scheme, "mle")["pars"])
    ll = {k: _weighted_log_likelihood(entry.klass, p, u, v, w) for k, p in optima.items()}
    best = _optimum(entry, u, v, w, list(optima.values()))
    for who, value in ll.items():
        assert 0.0 <= best - value <= SHORTFALL[who], (who, best - value)
    assert max(ll[r] for r in REFERENCES) - ll["pmcprg"] <= SHORTFALL["pmcprg"]


@pytest.mark.parametrize("dataset", WEIGHTED_SETS)
def test_reference_log_likelihood_definitions(dataset):
    """What each reference calls its (weighted) log-likelihood, against pmcprg's total Σ wᵢ log cᵢ.

    * VineCopula, MLE: the total (``logLik`` = Σ wᵢ log cᵢ);
    * VineCopula, itau: Σ log cᵢ over every row — the weights ignored, zero
      weights included;
    * pyvinecopulib: (n_rows/Σw)·Σ wᵢ log cᵢ — weights rescaled to mean 1 over
      the rows, zero-weight rows included — while ``nobs`` counts the positive
      weights: the {0, 1} fit reports 300/Σw times the subset's value.
    """
    entry, short = ENTRY[dataset], SHORT[dataset]
    tol = _student_aware(dataset)
    for scheme in SCHEMES:
        u, v, w, _ = _scheme(dataset, scheme)
        ones = np.ones(u.size)
        for method in ("itau", "mle"):
            for reference in REFERENCES:
                fit = _ref_fit(reference, dataset, scheme, method)
                if fit is None:
                    continue
                p = _params_of_key(short, fit["pars"])
                total = _weighted_log_likelihood(entry.klass, p, u, v, w)
                plain = _weighted_log_likelihood(entry.klass, p, u, v, ones)
                assert abs(fit["loglik_weighted_own"] - total) <= tol * abs(total)
                if reference == "VineCopula":
                    expected = total if method == "mle" else plain
                    assert fit["nobs"] == u.size
                else:
                    expected = u.size / w.sum() * total
                    assert fit["nobs"] == int(np.count_nonzero(w))
                assert abs(fit["loglik_reported"] - expected) <= tol * abs(expected), (
                    reference, scheme, method, fit["loglik_reported"], expected)


# ---------------------------------------------------------------------------
# {0, 1} weights and weight scaling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dataset", WEIGHTED_SETS)
def test_zero_one_weights_equal_the_subset_fit(dataset):
    """{0, 1} weights give pmcprg's unweighted fit on the subset (vinecopulib's own check).

    Bit-identical here (τ and every fitted parameter, both environments): the
    objective sums the positive-weight points only, the same arrays in the
    same order. Asserted to the scaling tolerances, which cover a BLAS that
    groups the τ's sums differently and so perturbs a two-parameter start.
    The references pass it too, at their own shortfalls
    (:func:`test_weighted_mle_reaches_the_optimum` finds the same maximum for
    ``w01`` and ``subset``).
    """
    entry = ENTRY[dataset]
    u, v, w, _ = _scheme(dataset, "w01")
    us, vs, ws, _ = _scheme(dataset, "subset")
    assert abs(_weighted_kendall_tau(u, v, w) - _weighted_kendall_tau(us, vs, ws)) <= TOL_TAU["pmcprg"]
    p01, psub = _pmcprg_fit(dataset, u, v, w), _pmcprg_fit(dataset, us, vs, ws)
    two = len(p01) > 1
    assert abs(p01["tau_k"] - psub["tau_k"]) <= TOL_SCALE["tau2" if two else "tau1"]
    l01 = _weighted_log_likelihood(entry.klass, p01, us, vs, ws)
    lsub = _weighted_log_likelihood(entry.klass, psub, us, vs, ws)
    assert abs(l01 - lsub) <= TOL_SCALE["loglik2" if two else "loglik1"]
    assert abs(_optimum(entry, u, v, w, [p01, psub]) - _optimum(entry, us, vs, ws, [p01, psub])) \
        <= TOL_SCALE["loglik1"]


@pytest.mark.parametrize("dataset", WEIGHTED_SETS)
def test_weight_scaling_invariance(dataset):
    """Multiplying every weight by c leaves pmcprg's estimate unchanged, down to Σw ≈ 1.3e-4."""
    entry = ENTRY[dataset]
    u, v, w, _ = _scheme(dataset, "gen")
    base = _pmcprg_fit(dataset, u, v, w)
    two = len(base) > 1
    ll0 = _weighted_log_likelihood(entry.klass, base, u, v, w)
    scales = (1 / 3, 7.0, 1e3, 1e6, 2.0 ** 10, 1e-2, 1e-4, 1e-6, 2.0 ** -20)
    for c in scales:
        p = _pmcprg_fit(dataset, u, v, w * c)
        ll = _weighted_log_likelihood(entry.klass, p, u, v, w)
        assert abs(p["tau_k"] - base["tau_k"]) <= TOL_SCALE["tau2" if two else "tau1"], (c, p, base)
        assert abs(ll - ll0) <= TOL_SCALE["loglik2" if two else "loglik1"], (c, ll - ll0)


# ---------------------------------------------------------------------------
# Selection: recovery, and against VineCopula's BiCopSelect
# ---------------------------------------------------------------------------

def _vc_selection(dataset: str, seed: int, weighting: str) -> dict:
    (rec,) = [s for s in _vc()["selection"]
              if (s["dataset"], int(s["seed"]), s["weighting"]) == (dataset, seed, weighting)]
    return rec


def _vc_short(code: int, candidates: list[str]) -> str:
    (short,) = [s for s in candidates if VC_CODE[s] == code]
    return short


def _candidates(dataset: str) -> list[str]:
    true = SHORT[dataset]
    return ["Prod", "Gauss"] + ([] if true == "Gauss" else [true])


def _check_selection(dataset: str, seeds) -> None:
    """Every decision, pmcprg's and VineCopula's, on the samples of ``seeds``; and the fits behind them."""
    cands = _candidates(dataset)
    true = SHORT[dataset]
    problems = []
    for r in _selection_rows():
        seed = int(r["seed"])
        if r["dataset"] != dataset or seed not in seeds:
            continue
        for weighting in ("unit", "formula"):
            u, v, w = _selection_sample(r, weighting)
            rec = _vc_selection(dataset, seed, weighting)
            # Each candidate's weighted MLE, pmcprg's against VineCopula's.
            _, fits = _copula_candidate_fits(cands, u, v, w, "mle")
            ours = {short: score for short, _, score in fits}
            for f in rec["fits"]:
                short = _vc_short(f["code"], cands)
                cls = CopulaEnum.from_short_name(short).klass
                theirs = _weighted_log_likelihood(cls, _vc_params(short, f["par"], f["par2"]), u, v, w)
                if abs(f["loglik"] - theirs) > _student_aware(short) * max(abs(theirs), 1.0):
                    problems.append((seed, weighting, short, "VineCopula's loglik", f["loglik"], theirs))
                if not (-TOL_SELECTION_LOGLIK["pmcprg"] <= ours[short] - theirs
                        <= TOL_SELECTION_LOGLIK["VineCopula"]):
                    problems.append((seed, weighting, short, "fit", ours[short] - theirs))
            # The decisions, as ICE makes them.
            for crit, vc_crit in CRITERIA.items():
                key = (dataset, seed, weighting, crit)
                pm = _select_and_fit_copula(cands, u, v, w, criterion=crit)["name"]
                vc = _vc_short(rec["selected"][vc_crit], cands)
                expect_pm = RECOVERY_FAILURES.get(key, true)
                expect_vc = SELECTION_DISAGREEMENTS.get(key, expect_pm)
                if (pm, vc) != (expect_pm, expect_vc):
                    problems.append((key, "pmcprg", pm, expect_pm, "VineCopula", vc, expect_vc))
    assert not problems, problems


@pytest.mark.parametrize("dataset", SELECTION_SETS)
def test_selection_recovery_fast(dataset):
    """Seeds 1–5 of each family: the true family among {Product, Gauss, true}, as VineCopula picks it."""
    _check_selection(dataset, FAST_SEEDS)


@pytest.mark.slow
@pytest.mark.parametrize("dataset", SELECTION_SETS)
def test_selection_recovery_all_seeds(dataset):
    """Seeds 6–20: the rest of the documented recovery study (module docstring)."""
    _check_selection(dataset, SLOW_SEEDS)


def test_selection_disagreement_is_the_bic_sample_size():
    """pmcprg's own log-likelihoods, charged VineCopula's log n per parameter, make VineCopula's choice."""
    for (dataset, seed, weighting, crit), vc_pick in SELECTION_DISAGREEMENTS.items():
        assert crit == "bic"
        (r,) = [r for r in _selection_rows() if (r["dataset"], int(r["seed"])) == (dataset, seed)]
        u, v, w = _selection_sample(r, weighting)
        cands = _candidates(dataset)
        _, fits = _copula_candidate_fits(cands, u, v, w, "mle")
        k = {s: CopulaEnum.from_short_name(s).klass.n_params for s in cands}

        def pick(penalty):
            return max(cands, key=lambda s: 2.0 * dict((f[0], f[2]) for f in fits)[s] - penalty * k[s])

        ours = _select_and_fit_copula(cands, u, v, w, criterion=crit)["name"]
        assert ours == pick(math.log(w.sum())) != vc_pick
        assert pick(math.log(u.size)) == vc_pick
        # VineCopula's own log-likelihoods lead to the same choice under either penalty's owner.
        rec = _vc_selection(dataset, seed, weighting)
        vc_ll = {_vc_short(f["code"], cands): f["loglik"] for f in rec["fits"]}
        assert max(cands, key=lambda s: 2.0 * vc_ll[s] - math.log(u.size) * k[s]) == vc_pick
        assert max(cands, key=lambda s: 2.0 * vc_ll[s] - math.log(w.sum()) * k[s]) == ours


def test_selection_penalties_match_vinecopula_parameter_counts():
    """AIC 2k − 2ℓ and BIC k·log Σw − 2ℓ (as scores, negated), with VineCopula's parameter counts."""
    rec = _vc()["selection"]
    npars = {}
    for s in rec:
        for f in s["fits"]:
            npars[f["code"]] = f["npars"]
    u, v, w = _selection_sample(_selection_rows()[0], "formula")
    params = {"Prod": {"tau_k": 0.0}, "Clayton90": {"tau_k": -0.3},
              "Student": {"tau_k": 0.3, "df": 5.0}, "BB1": {"tau_k": 0.3, "delta": 1.2}}
    for short, code in VC_CODE.items():
        entry = CopulaEnum.from_short_name(short)
        cls = entry.klass
        assert cls.n_params == npars[code], (short, cls.n_params, npars[code])
        p = params.get(short, {"tau_k": 0.3})
        ll = _weighted_log_likelihood(cls, p, u, v, w)
        assert np.isfinite(ll)
        assert _score_aic(cls, entry, p, u, v, w) == 2.0 * ll - 2.0 * npars[code]
        assert _score_bic(cls, entry, p, u, v, w) == 2.0 * ll - npars[code] * np.log(w.sum())


# ---------------------------------------------------------------------------
# The public API is the same engine (FR-12)
# ---------------------------------------------------------------------------
#
# ``CopulaVirt.fit(uv, method, weights=w, pseudo_obs=True)`` runs ICE's
# functions above for generic weights, bit for bit; unit and {0, 1} weights
# are the unweighted fit of the kept rows (bit for bit,
# ``test_fr12_weighted_fit.py``), whose one-parameter search stops at Brent's
# default ``xatol`` 1e-5 and whose Student/BB1 optimisers are their own:
# measured 9.3e-10 nat below the polished maximum at most (Clayton 90°, unit);
# Student under scipy < 1.17 (its t quantile, and L-BFGS-B before its C port):
# 5.96e-9 on the Linux minimum-versions job (scipy 1.10), below 5e-9 on macOS.
SHORTFALL_UNWEIGHTED_PATH = 5e-9
SHORTFALL_UNWEIGHTED_PATH_STUDENT = 3e-8 if _SCIPY < (1, 17) else SHORTFALL_UNWEIGHTED_PATH

# itau of Student: ν by a Brent search in 1/ν at ρ(τ̂) (``xatol`` 1e-6) — a
# profile maximum, as pyvinecopulib's itau finds it. Measured against
# pyvinecopulib: ν 1.0e-6 relative, the log-likelihood at pmcprg's τ̂ within
# 7.8e-13 nat either way (the same maximum). VineCopula searches ν in [2, 10]
# at tol = 1: up to 5.4e-2 nat below (the {0, 1} scheme, ν̂ 9.44 against 11.89).
TOL_ITAU_NU_REL = 1e-5
TOL_ITAU_PROFILE = 1e-11


def _entry_of(dataset: str):
    return ENTRY.get(dataset, CopulaEnum.GAUSSIAN)        # the tie grid is Gaussian


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("dataset", WEIGHTED_SETS)
def test_public_weighted_mle_is_the_engine(dataset, scheme):
    entry = ENTRY[dataset]
    cls = entry.klass
    u, v, w, _ = _scheme(dataset, scheme)
    r = cls.fit(np.column_stack([u, v]), method="mle", weights=w, pseudo_obs=True)
    engine = _pmcprg_fit(dataset, u, v, w)
    params = dict(r.copula.params)
    ll = _weighted_log_likelihood(cls, params, u, v, w)
    if scheme == "gen":
        # Generic weights: ICE's weighted MLE itself, and its log-likelihood.
        assert r.weights is not None and params == engine
        assert r.log_likelihood == ll and r.converged
    else:
        # Unit or {0, 1} weights: the unweighted fit of the kept rows.
        assert r.weights is None and r.n_obs == int(np.count_nonzero(w))
        best = _optimum(entry, u, v, w, [engine, params])
        tol = SHORTFALL_UNWEIGHTED_PATH_STUDENT if dataset == "student" else SHORTFALL_UNWEIGHTED_PATH
        assert 0.0 <= best - ll <= tol, best - ll


@pytest.mark.parametrize("dataset", list(_weighted_data()))
def test_public_itau_inverts_the_references_weighted_tau(dataset):
    """``fit(method='tau', weights=…)`` inverts the weighted τ-b — the tie grid included."""
    entry = _entry_of(dataset)
    lo, hi = entry.constructible_tau_range()
    for scheme in SCHEMES:
        u, v, w, _ = _scheme(dataset, scheme)
        r = entry.klass.fit(np.column_stack([u, v]), method="tau", weights=w, pseudo_obs=True)
        for reference in REFERENCES:
            ref = float(np.clip(_ref_kendall(reference, dataset, scheme), lo, hi))
            assert abs(r.tau_k - ref) <= TOL_TAU["pmcprg"] + TOL_TAU[reference], (scheme, reference)


def test_public_student_itau_nu_is_the_profile_maximum():
    """Student's itau ν is pyvinecopulib's (weighted) profile maximum; VineCopula stops short."""
    entry = ENTRY["student"]
    for scheme in SCHEMES:
        u, v, w, _ = _scheme("student", scheme)
        r = entry.klass.fit(np.column_stack([u, v]), method="tau", weights=w, pseudo_obs=True)
        assert r.method == "tau" and r.converged
        ours = _weighted_log_likelihood(entry.klass, dict(r.copula.params), u, v, w)
        pv = _params_of_key("Student", _ref_fit("pyvinecopulib", "student", scheme, "itau")["pars"])
        vc = _params_of_key("Student", _ref_fit("VineCopula", "student", scheme, "itau")["pars"])
        assert abs(r.tau_k - pv["tau_k"]) <= TOL_ITAU
        assert abs(r.copula.df - pv["df"]) <= TOL_ITAU_NU_REL * pv["df"], (scheme, r.copula.df)
        ll_pv = _weighted_log_likelihood(entry.klass, {"tau_k": r.tau_k, "df": pv["df"]}, u, v, w)
        ll_vc = _weighted_log_likelihood(entry.klass, {"tau_k": r.tau_k, "df": vc["df"]}, u, v, w)
        assert abs(ours - ll_pv) <= TOL_ITAU_PROFILE, (scheme, ours - ll_pv)
        assert ours - ll_vc >= -TOL_ITAU_PROFILE, (scheme, ours - ll_vc)


@pytest.mark.parametrize("dataset", SELECTION_SETS)
def test_public_fit_best_selects_as_the_engine(dataset):
    """``fit_best(method='mle', weights=w, criterion=…)`` makes ICE's decisions (seeds 1–5).

    With the deterministic weights the winner's parameters are the engine's bit
    for bit; with unit weights the fits are the unweighted ones and only the
    decisions are compared (all agree).
    """
    cands = _candidates(dataset)
    classes = [CopulaEnum.from_short_name(s).klass for s in cands]
    problems = []
    for r in _selection_rows():
        seed = int(r["seed"])
        if r["dataset"] != dataset or seed not in FAST_SEEDS:
            continue
        for weighting in ("unit", "formula"):
            u, v, w = _selection_sample(r, weighting)
            for crit, ice_crit in (("aic", "aic"), ("bic", "bic"), ("loglik", "mle")):
                fb = CopulaVirt.fit_best(np.column_stack([u, v]), families=classes, method="mle",
                                         weights=w, criterion=crit, pseudo_obs=True)
                blk = _select_and_fit_copula(cands, u, v, w, criterion=ice_crit)
                best = fb[0].copula
                if best.copula_enum.value.SHORT_NAME != blk["name"]:
                    problems.append((seed, weighting, crit, best.copula_enum.value.SHORT_NAME,
                                     blk["name"]))
                elif weighting == "formula" and (
                        best.params["tau_k"] != blk["tau"]
                        or any(best.params[k] != val for k, val in blk.items()
                               if k not in ("name", "tau"))):
                    problems.append((seed, weighting, crit, "parameters", dict(best.params), blk))
    assert not problems, problems
