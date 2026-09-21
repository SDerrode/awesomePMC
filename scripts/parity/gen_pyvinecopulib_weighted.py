#!/usr/bin/env python3
"""Weighted-estimation references from pyvinecopulib (audit FR-11, wave 3).

Reads the fixed samples of ``pmcprg/tests/data/parity/weighted_data.csv``
(``gen_weighted_data.py``) and records, for each sample and each weighting —

* ``unit``: no weights (the unweighted fit);
* ``gen``: the generic positive weights w = w_gen/2²⁰;
* ``w01``: the {0, 1} weights;
* ``subset``: no weights, on the points of weight 1 only (vinecopulib's own
  check: {0, 1} weights must give the fit on the subset) —

the weighted Kendall τ of ``pyvinecopulib.utils.wdm`` and the fits of
``Bicop.fit`` with ``parametric_method`` ``"itau"`` (one-parameter families
and Student) and ``"mle"``, the family and rotation held fixed: the native
parameters, the key's parameters (README), ``Bicop.tau``, the log-likelihood
the fitted object reports (``Bicop.loglik()``, which pyvinecopulib stores at
fit time) and the raw weighted sum Σ wᵢ log c(uᵢ, vᵢ) of the reference's own
density at its optimum. The sample with ties (``gaussian_ties``) has its τ
only. Written, with the provenance and the md5 of the data file, to
``pmcprg/tests/data/parity/weighted_pyvinecopulib.json``, read by
``pmcprg/tests/test_parity_weighted.py``. It never imports ``pmcprg``.

Usage, from the repository root::

    .venv-parity/bin/python scripts/parity/gen_pyvinecopulib_weighted.py
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import json
import platform
import re
import sys
from pathlib import Path

import numpy as np
import pyvinecopulib as pv
from pyvinecopulib.utils import wdm

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "pmcprg" / "tests" / "data" / "parity" / "weighted_data.csv"
OUT = ROOT / "pmcprg" / "tests" / "data" / "parity" / "weighted_pyvinecopulib.json"
W_SCALE = 2 ** 20

ITAU = {"gaussian", "student", "clayton", "gumbel", "frank"}   # not BB1 (pyvinecopulib: 1-par + t)


def read_data() -> list[dict]:
    with open(DATA, newline="") as fh:
        rows = list(csv.DictReader(line for line in fh if not line.startswith("#")))
    out = []
    for r in rows:
        den = int(r["denom"])
        ints = {k: np.array([int(x) for x in r[k].split()], dtype=np.int64)
                for k in ("u_int", "v_int", "w_gen", "w01")}
        out.append({
            "dataset": r["dataset"], "family": r["family"], "rotation": int(r["rotation"]),
            "u": ints["u_int"] / den, "v": ints["v_int"] / den,
            "w_gen": ints["w_gen"] / W_SCALE, "w01": ints["w01"].astype(float),
        })
    return out


def schemes(d: dict):
    """(scheme, u, v, weights or None) — None: unweighted."""
    keep = d["w01"] > 0
    return [("unit", d["u"], d["v"], None), ("gen", d["u"], d["v"], d["w_gen"]),
            ("w01", d["u"], d["v"], d["w01"]), ("subset", d["u"][keep], d["v"][keep], None)]


def key_pars(family: str, rotation: int, native: list[float]) -> list[float]:
    """pyvinecopulib's parameters are the key's (README): rotation passed separately, θ > 0."""
    return list(native)


def fit(d: dict, u, v, w, method: str) -> dict:
    fam = getattr(pv.BicopFamily, d["family"])
    ctl = pv.FitControlsBicop(family_set=[fam], parametric_method=method,
                              weights=np.array([]) if w is None else np.ascontiguousarray(w))
    b = pv.Bicop(family=fam, rotation=d["rotation"])
    uv = np.asfortranarray(np.column_stack((u, v)))
    b.fit(uv, controls=ctl)
    if b.family != fam or b.rotation != d["rotation"]:
        sys.exit(f"{d['dataset']}: the fit changed the family or the rotation")
    native = [float(x) for x in np.asarray(b.parameters).ravel()]
    logc = np.log(b.pdf(uv))
    ww = np.ones(u.size) if w is None else w
    return {
        "method": method,
        "native": f"Bicop(family=BicopFamily.{d['family']}, rotation={d['rotation']}, "
                  f"parameters=[{', '.join(repr(x) for x in native)}])",
        "pars": key_pars(d["family"], d["rotation"], native),
        "tau": float(b.tau),
        "loglik_reported": float(b.loglik()),
        "loglik_weighted_own": float(np.dot(ww, logc)),
        "loglik_unweighted_own": float(np.sum(logc)),
        "sum_w": float(np.sum(ww)),
        "nobs": int(b.nobs),
    }


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> int:
    data = read_data()
    fits, taus = [], {}
    for d in data:
        for scheme, u, v, w in schemes(d):
            taus.setdefault(d["dataset"], {})[scheme] = float(
                wdm(u, v, "kendall", np.array([]) if w is None else w))
            if d["dataset"] == "gaussian_ties":
                continue
            for method in ("itau", "mle"):
                if method == "itau" and d["family"] not in ITAU:
                    continue
                rec = {"dataset": d["dataset"], "scheme": scheme}
                rec.update(fit(d, u, v, w, method))
                fits.append(rec)

    version = re.search(r'__version__\s*=\s*"([^"]+)"',
                        (ROOT / "pmcprg" / "__init__.py").read_text()).group(1)
    doc = {
        "format": 1,
        "about": ("FR-11 weighted-estimation references, generated by "
                  "scripts/parity/gen_pyvinecopulib_weighted.py - do not edit by hand. Floats "
                  "are shortest round-trip repr of IEEE-754 doubles."),
        "reference": {"package": "pyvinecopulib", "version": pv.__version__,
                      "backend": "vinecopulib (C++), bundled; weighted Kendall tau from wdm"},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform(), "machine": platform.machine()},
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "repository_version": version,
        "command": ".venv-parity/bin/python scripts/parity/gen_pyvinecopulib_weighted.py",
        "data": {"file": "pmcprg/tests/data/parity/weighted_data.csv", "md5": md5(DATA)},
        "definitions": {
            "scheme": "unit: unweighted; gen: w = w_gen/2^20; w01: the {0,1} weights; subset: "
                      "unweighted on the points with w01 = 1",
            "kendall": "pyvinecopulib.utils.wdm(u, v, 'kendall', weights)",
            "fit": "Bicop(family, rotation).fit(uv, FitControlsBicop(family_set=[family], "
                   "parametric_method=method, weights=w)); family and rotation held fixed",
            "pars": "reference-agnostic key (README): [rho], [rho, nu], [theta] of the "
                    "unrotated family (theta > 0 for every rotation), [theta, delta]",
            "tau": "Bicop.tau of the fitted parameters (sign flipped for 90/270)",
            "loglik_reported": "Bicop.loglik() after the fit (the value stored at fit time)",
            "loglik_weighted_own": "sum_i w_i log Bicop.pdf(u_i, v_i) (w = 1 unweighted)",
            "loglik_unweighted_own": "sum_i log Bicop.pdf(u_i, v_i)",
            "sum_w": "sum_i w_i (n unweighted)",
            "nobs": "Bicop.nobs",
        },
        "kendall": taus,
    }
    with open(OUT, "w") as fh:
        head = json.dumps(doc, indent=1, ensure_ascii=False)
        fh.write(head[:-2] + ',\n "fits": [\n')
        fh.write(",\n".join("  " + json.dumps(c, ensure_ascii=False) for c in fits))
        fh.write("\n ]\n}\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(fits)} fits, {len(taus)} data sets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
