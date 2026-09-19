"""Why the Markov LR test is too wide on HMC-DN at N = 500 — diagnosis (P6).

The study (``lr_study.py``) measured, on HMC-DN, K = 2, ``"state-markov"``
against a common Markov mask, N = 500, 200 null replications: size 0.135
(χ²) and 0.175 (warp-speed bootstrap), 9 negative statistics (min −1.42),
95 % quantile 8.05. At the time the statistic was the difference of the two
ICE fixed points, 2 (LL1(θ̂1, φ̂1) − LL0(θ̂0)). This script takes the same
replications (same seeds) apart.

``--decompose`` (every null replication of a cell; ``results/lr_decompose_<cell>.csv``)
    Refits both models as the test did and evaluates, per replication:

    * ``LL0_th1`` — the null at the alternative's θ: LL_ignorable(θ̂1) +
      log p(m | common mask MLE). ``2 (LL0_th1 − ll0)`` is the part of the
      fits' statistic due to θ̂1 ≠ θ̂0 alone ("θ-part");
    * ``P0`` / ``P1`` — the log-likelihood maximised over the alternative's
      mechanism with θ fixed at θ̂0 / θ̂1 (L-BFGS-B on the logits, the exact
      gradient by Fisher's identity: one forward-backward pass per
      evaluation, no boundary guard);
    * ``L1step0`` — one guarded mechanism M-step at θ̂0;
    * the ICE traces: iterations, the largest log-likelihood on each path.

``--mle`` (listed replications; ``results/lr_mle_<cell>.csv``)
    Maximises the observed-data log-likelihood directly over all the
    parameters of the HMC-DN model (K = 2: A, two Gaussian margins, four
    Gaussian copulas — 10 parameters; + 4 for the Markov mechanism),
    L-BFGS-B with forward-difference gradients (step 1e-6 on the
    unconstrained scale), from several starts: the null from θ̂0, θ̂1 and the
    true θ; the alternative from (θ̂1, φ̂1), ICE's best iterate, and the null
    maximum with the common mask. LR_mle = 2 (max alt − max null), the
    alternative never below the null.

Run from the repository root::

    .venv/bin/python report/missing_state/lr_diagnosis.py --decompose --jobs 6
    .venv/bin/python report/missing_state/lr_diagnosis.py --mle --jobs 5 \\
        --r 111,78,197,11,198,89,7,143,$(seq -s, 0 29)

Measured (2026-09, loaded Apple arm64 laptop): ``--decompose`` 5 min on 5
processes; ``--mle`` 15 min on 5 processes for 36 replications (~3 min per
replication). The numbers are in ``README.md``, "Diagnosis".
"""
from __future__ import annotations

import argparse
import copy
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = HERE / "results"
PAIRS = ((0, 0), (0, 1), (1, 0), (1, 1))


# ---------------------------------------------------------------------------
# Profile over the mechanism, θ fixed (exact: L-BFGS-B, Fisher's identity)
# ---------------------------------------------------------------------------

def profile_phi(model, Yn, miss, gn, mech0, kind):
    """max_φ log p(y_obs, m | θ, φ), θ of ``model`` fixed, from ``mech0``."""
    from scipy.optimize import minimize

    from pmcprg.pmc import gaps
    from pmcprg.pmc.missingness import StateMarkovMissingness, StateMissingness
    K = model.K
    prev, cur = miss[:-1], miss[1:]
    m0 = float(miss[0])
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))                    # noqa: E731
    nev = [0]

    def f(z):
        nev[0] += 1
        z = np.clip(z, -30, 30)
        if kind == "state":
            p = sig(z)
            mech = StateMissingness(rates=tuple(p))
        else:
            a, b = sig(z[:K]), sig(z[K:])
            mech = StateMarkovMissingness(onset=tuple(a), persistence=tuple(b))
        post = gaps._posterior(model.with_missingness(mech), Yn, gn, False)[0]
        g = post.gamma
        if kind == "state":
            grad = (g * (miss[:, None] - p[None, :])).sum(0)
        else:
            gn1 = g[1:]
            gu = (gn1[~prev] * (cur[~prev][:, None] - a[None, :])).sum(0)
            gv = (gn1[prev] * (cur[prev][:, None] - b[None, :])).sum(0)
            den = 1 - b + a
            gu = gu + g[0] * (m0 * (1 - a) - a * (1 - a) / den)
            gv = gv + g[0] * (-(1 - m0) * b + b * (1 - b) / den)
            grad = np.concatenate([gu, gv])
        return -post.log_lik, -grad

    def logit(p):
        p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
        return np.log(p) - np.log1p(-p)
    if kind == "state":
        z0 = logit(mech0.rates)
    else:
        z0 = np.concatenate([logit(mech0.onset), logit(mech0.persistence)])
    f0 = f(z0)[0]
    res = minimize(f, z0, jac=True, method="L-BFGS-B",
                   options={"ftol": 1e-14, "gtol": 1e-7, "maxiter": 300})
    return -min(res.fun, f0), nev[0], res


def _data(model_name, scen, N, r):
    import lr_study as S

    from pmcprg.pmc import simulate
    from pmcprg.pmc.gaps import missing_mask
    from pmcprg.pmc.missingness_lr import _fit_cfg
    mech, params, (alt, null), role = S.SCENARIOS[scen]
    m = S._model(model_name)
    X, Y = simulate(m, N=N, seed=S._seed("sim", model_name, scen, N, r))
    Yn = S._mask(Y, X, mech, params, S._seed("mask", model_name, scen, N, r))
    return m, Yn, missing_mask(Yn), _fit_cfg(m, S.ICE_CFG[model_name]), alt, null


def job_decompose(args):
    model_name, scen, N, r = args
    import logging as _lg
    _lg.disable(_lg.WARNING)
    from pmcprg.pmc import gaps
    from pmcprg.pmc.ice import _parse_ice_cfg, ice
    from pmcprg.pmc.missingness import (StateMarkovMissingness, common_mechanism,
                                        estimate_mechanism, mask_log_likelihood)
    from pmcprg.pmc.missingness_lr import _log_lik
    t0 = time.time()
    m, Yn, miss, cfg, alt, null = _data(model_name, scen, N, r)
    gn = _parse_ice_cfg(m, cfg).get("gap_nodes")
    row = dict(model=model_name, scenario=scen, N=N, r=r, M=int(miss.sum()),
               bursts=int(miss[0] + np.sum(~miss[:-1] & miss[1:])))
    if null == "common":
        fit0, tr0 = ice(m, Yn, {**cfg, "missingness": "ignorable"})
        mc = common_mechanism(alt, miss, m.K)
        mll = mask_log_likelihood(mc, miss)
        null_model = fit0.with_missingness(mc)
        ll0 = _log_lik(fit0, Yn, miss, cfg) + mll
    else:
        null_model, tr0 = ice(m, Yn, {**cfg, "missingness": "state"})
        ll0 = _log_lik(null_model, Yn, miss, cfg)
        mc, mll = null_model.missingness, float("nan")
    fit1, tr1 = ice(null_model, Yn, {**cfg, "missingness": alt, "init": "model"})
    ll1 = _log_lik(fit1, Yn, miss, cfg)
    row.update(LR=2 * (ll1 - ll0), ll0=ll0, ll1=ll1, it0=len(tr0.log_liks), it1=len(tr1.log_liks),
               tr0_maxgap=max(tr0.log_liks) - tr0.log_liks[-1],
               tr1_max=max(tr1.log_liks) - ll0, tr1_argmax=int(np.argmax(tr1.log_liks)),
               tr1_first=tr1.log_liks[0] - ll0)
    if null == "common":
        row["LL0_th1"] = _log_lik(fit1.with_missingness(None), Yn, miss, cfg) + mll
    else:
        row["LL0_th1"] = profile_phi(fit1, Yn, miss, gn, null_model.missingness, "state")[0]
    start0 = mc if alt == mc.mechanism else StateMarkovMissingness(onset=mc.rates,
                                                                   persistence=mc.rates)
    P0, nev0, _ = profile_phi(null_model, Yn, miss, gn, start0, alt)
    P1, nev1, _ = profile_phi(fit1, Yn, miss, gn, fit1.missingness, alt)
    g0 = gaps._posterior(null_model, Yn, gn, False)[0].gamma
    L1step = _log_lik(null_model.with_missingness(estimate_mechanism(alt, g0, miss)),
                      Yn, miss, cfg)
    row.update(P0=P0, P1=P1, L1step0=L1step, nev0=nev0, nev1=nev1)

    def th(mdl):
        return ([b["params"]["loc"] for b in mdl.raw["margins"]]
                + [b["params"]["scale"] for b in mdl.raw["margins"]]
                + [mdl.raw["prior"]["A"][0][0], mdl.raw["prior"]["A"][1][1]])
    t0v, t1v = th(null_model), th(fit1)
    for k, name in enumerate(("mu0", "mu1", "s0", "s1", "A00", "A11")):
        row[f"th0_{name}"], row[f"th1_{name}"] = t0v[k], t1v[k]
    tb = fit1.missingness.to_table()
    for k in ("rates", "onset", "persistence"):
        for i, v in enumerate(tb.get(k, ())):
            row[f"alt_{k}{i}"] = v
    row["seconds"] = time.time() - t0
    return row


# ---------------------------------------------------------------------------
# Direct maximisation (HMC-DN, K = 2, Gaussian margins and copulas)
# ---------------------------------------------------------------------------

def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def theta_vec(model):
    raw = model.raw
    A = raw["prior"]["A"]
    mg = {b["i"]: b["params"] for b in raw["margins"]}
    cop = {(c["i"], c["j"]): c["tau"] for c in raw["copulas"]}
    return np.array([_logit(A[0][1]), _logit(A[1][0]), mg[0]["loc"], mg[1]["loc"],
                     np.log(mg[0]["scale"]), np.log(mg[1]["scale"])]
                    + [np.arctanh(np.clip(cop[p], -0.999999, 0.999999)) for p in PAIRS])


def phi_vec(mech):
    return np.concatenate([_logit(mech.onset), _logit(mech.persistence)])


def build(template_raw, z, with_phi):
    from pmcprg.pmc import PMCModel
    from pmcprg.pmc.missingness import StateMarkovMissingness
    raw = copy.deepcopy(template_raw)
    raw.pop("missingness", None)
    z = np.clip(z, -30, 30)
    a01, a10 = _sig(z[0]), _sig(z[1])
    raw["prior"]["A"] = [[1 - a01, a01], [a10, 1 - a10]]
    for b in raw["margins"]:
        b["params"] = {"loc": float(z[2 + b["i"]]), "scale": float(np.exp(z[4 + b["i"]]))}
    for c in raw["copulas"]:
        c["tau"] = float(np.tanh(z[6 + PAIRS.index((c["i"], c["j"]))]))
        c["name"] = "Gauss"
    m = PMCModel.from_dict(raw)
    if with_phi:
        m = m.with_missingness(StateMarkovMissingness(onset=tuple(_sig(z[10:12])),
                                                      persistence=tuple(_sig(z[12:14]))))
    return m


def maximise(fun, z0, h=1e-6):
    from scipy.optimize import minimize
    nev = [0]

    def f(z):
        nev[0] += 1
        try:
            v = fun(z)
        except Exception:
            return 1e10
        return -v if np.isfinite(v) else 1e10

    def fg(z):
        f0 = f(z)
        g = np.empty_like(z)
        for k in range(z.size):
            zz = z.copy()
            zz[k] += h
            g[k] = (f(zz) - f0) / h
        return f0, g
    f_start = f(z0)
    res = minimize(fg, z0, jac=True, method="L-BFGS-B",
                   options={"ftol": 1e-13, "gtol": 1e-4, "maxiter": 400, "maxcor": 20})
    if res.fun > f_start:
        return -f_start, z0, nev[0]
    return -res.fun, res.x, nev[0]


def job_mle(args):
    model_name, scen, N, r = args
    import logging as _lg
    _lg.disable(_lg.WARNING)
    from pmcprg.pmc.ice import ice
    from pmcprg.pmc.missingness import common_mechanism, mask_log_likelihood
    from pmcprg.pmc.missingness_lr import _log_lik
    t0 = time.time()
    m, Yn, miss, cfg, alt, null = _data(model_name, scen, N, r)
    assert model_name == "hmc_dn" and alt == "state-markov" and null == "common"
    fit0, _ = ice(m, Yn, {**cfg, "missingness": "ignorable"})
    mc = common_mechanism(alt, miss, 2)
    mll = mask_log_likelihood(mc, miss)
    ll0 = _log_lik(fit0, Yn, miss, cfg) + mll
    null_model = fit0.with_missingness(mc)
    fit1, _ = ice(null_model, Yn, {**cfg, "missingness": alt, "init": "model"})
    ll1 = _log_lik(fit1, Yn, miss, cfg)
    best, _ = ice(null_model, Yn, {**cfg, "missingness": alt, "init": "model",
                                   "return_best_iterate": True})
    llb = _log_lik(best, Yn, miss, cfg)
    tmpl = fit0.raw
    f_null = lambda z: _log_lik(build(tmpl, z, False), Yn, miss, cfg) + mll    # noqa: E731
    f_alt = lambda z: _log_lik(build(tmpl, z, True), Yn, miss, cfg)            # noqa: E731
    row = dict(model=model_name, scenario=scen, N=N, r=r, M=int(miss.sum()),
               LR=2 * (ll1 - ll0), ll0=ll0, ll1=ll1, ll_best=llb)
    nulls = {}
    for tag, mdl in (("th0", fit0), ("th1", fit1), ("true", m)):
        nulls[tag] = maximise(f_null, theta_vec(mdl))
        row[f"n_{tag}"] = nulls[tag][0]
    tag0 = max(nulls, key=lambda k: nulls[k][0])
    L0, z0star = nulls[tag0][0], nulls[tag0][1]
    alts = {}
    for tag, zs in (("th1", np.concatenate([theta_vec(fit1), phi_vec(fit1.missingness)])),
                    ("thb", np.concatenate([theta_vec(best), phi_vec(best.missingness)])),
                    ("n*", np.concatenate([z0star, phi_vec(mc)]))):
        alts[tag] = maximise(f_alt, zs)
        row[f"a_{tag}"] = alts[tag][0]
    L1 = max(max(v[0] for v in alts.values()), L0)
    row.update(L0_mle=L0, L1_mle=L1, LR_mle=2 * (L1 - L0), null_best=tag0,
               alt_best=max(alts, key=lambda k: alts[k][0]))
    z1 = alts[row["alt_best"]][1]
    row.update(mle_a0=_sig(z1[10]), mle_a1=_sig(z1[11]), mle_b0=_sig(z1[12]), mle_b1=_sig(z1[13]),
               mle1_mu0=z1[2], mle1_mu1=z1[3], mle0_mu0=z0star[2], mle0_mu1=z0star[3])
    row["seconds"] = time.time() - t0
    return row


def _run(fn, jobs, workers, path):
    OUT.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(fn, j) for j in jobs]
        for k, fu in enumerate(as_completed(futs), 1):
            rows.append(fu.result())
            if k % 25 == 0 or k == len(futs):
                print(f"  {k}/{len(futs)} ({time.time() - t0:.0f} s)", flush=True)
    rows.sort(key=lambda d: d["r"])
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--decompose", action="store_true")
    ap.add_argument("--mle", action="store_true")
    ap.add_argument("--model", default="hmc_dn")
    ap.add_argument("--scenario", default="markov-null")
    ap.add_argument("--N", type=int, default=500)
    ap.add_argument("--R", type=int, default=200, help="--decompose: replications 0..R-1")
    ap.add_argument("--r", default="", help="--mle: comma-separated replications")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    cell = f"{args.model}_{args.scenario}_{args.N}"
    if args.decompose:
        jobs = [(args.model, args.scenario, args.N, r) for r in range(args.R)]
        _run(job_decompose, jobs, args.jobs, OUT / f"lr_decompose_{cell}.csv")
    if args.mle:
        rs = [int(x) for x in args.r.split(",") if x]
        _run(job_mle, [(args.model, args.scenario, args.N, r) for r in rs], args.jobs,
             OUT / f"lr_mle_{cell}.csv")


if __name__ == "__main__":
    main()
