#!/usr/bin/env python3
"""Interior parity references from pyvinecopulib for the families beyond the pilot (audit FR-11).

Of the families of ``scripts/parity/cases_extra.csv`` pyvinecopulib 1.0.0 has
one: the asymmetric logistic Tawn model, ``BicopFamily.tawn``, with the
three parameters ``[psi1, psi2, theta]`` and both weights free. This script
evaluates pdf, cdf, both h-functions, both h-inverses, Kendall's τ and the
two diagonal tail-dependence coefficients of the Tawn cases on the interior
points of ``scripts/parity/points.csv`` and writes them, with their
provenance, to ``pmcprg/tests/data/parity/extra_pyvinecopulib.json``, read by
``pmcprg/tests/test_parity_extra.py``.

Which native parameter is which weight is *measured*, not assumed: the
reference's CDF is compared with the textbook formula of the
reference-agnostic key (``tawn [theta, psi_u, psi_v]``, ψ_u the weight of
−log u; ``scripts/parity/README.md``) under both assignments, and the one
that matches is recorded under ``convention_checks.tawn_weights``; the
h-function conventions are measured as in ``gen_pyvinecopulib.py``, whose
helpers this script reuses. It never imports ``pmcprg``.

Usage, from the repository root::

    .venv-parity/bin/python scripts/parity/gen_pyvinecopulib_extra.py
"""

from __future__ import annotations

import csv
import datetime
import json
import platform
import re
import sys
from pathlib import Path

import numpy as np
import pyvinecopulib as pv

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gen_pyvinecopulib as base  # noqa: E402  (the pilot generator's helpers)

ROOT = HERE.parents[1]
OUT = ROOT / "pmcprg" / "tests" / "data" / "parity" / "extra_pyvinecopulib.json"

# Same as gen_r_extra.R: a wrong assignment of the weights is off by
# O(0.001) to O(1) at most points.
PARAM_TOL = 1e-9


def read_grid() -> list[list]:
    """Every case of ``cases_extra.csv``, as ``[family, rotation, pars]`` (the tests read it)."""
    with open(HERE / "cases_extra.csv", newline="") as fh:
        return [[r["family"], int(r["rotation"]),
                 [float(r[k]) for k in ("par1", "par2", "par3") if r[k]]]
                for r in csv.DictReader(fh)]


def read_tawn_cases() -> list[list[float]]:
    return [pars for family, _, pars in read_grid() if family == "tawn"]


def textbook_cdf(pars: list[float], U: np.ndarray) -> np.ndarray:
    """C = exp(−ℓ), ℓ = (1 − ψ_u)w + (1 − ψ_v)z + ((ψ_u w)^θ + (ψ_v z)^θ)^{1/θ}."""
    th, pu, pv_ = pars
    w, z = -np.log(U[:, 0]), -np.log(U[:, 1])
    return np.exp(-((1 - pu) * w + (1 - pv_) * z + ((pu * w) ** th + (pv_ * z) ** th) ** (1 / th)))


def native(pars: list[float], U: np.ndarray) -> tuple[list[float], str, float]:
    """The native ``[psi1, psi2, theta]`` whose CDF is the case's textbook CDF."""
    th, pu, pv_ = pars
    tb = textbook_cdf(pars, U)
    hits = []
    for order, label in (([pu, pv_], "psi1 = psi_u, psi2 = psi_v"),
                         ([pv_, pu], "psi1 = psi_v, psi2 = psi_u")):
        p = order + [th]
        b = pv.Bicop(family=pv.BicopFamily.tawn, parameters=np.array([[x] for x in p]))
        err = float(np.median(np.abs(b.cdf(U) - tb)))
        if err < PARAM_TOL:
            hits.append((p, label, err))
    if pu != pv_ and len(hits) != 1:
        sys.exit(f"tawn {pars}: no unique weight assignment matches")
    return hits[0]


def main() -> int:
    U = base.read_points()
    out_cases = []
    checks = {"fd_step": base.FD_STEP, "fd_tolerance": base.FD_TOL,
              "roundtrip_tolerance": base.ROUNDTRIP_TOL, "param_tolerance": PARAM_TOL,
              "judged_on": "median over the points (max recorded); param_* absolute against "
                           "the textbook CDF; pdf errors relative above pdf = 1, absolute below",
              "per_family": {}, "tawn_weights": None}
    worst = checks["per_family"].setdefault("tawn/0", {})
    for pars in read_tawn_cases():
        p, label, perr = native(pars, U)
        if checks["tawn_weights"] not in (None, label):
            sys.exit("tawn: the weight assignment depends on the case")
        checks["tawn_weights"] = label
        b = pv.Bicop(family=pv.BicopFamily.tawn, parameters=np.array([[x] for x in p]))
        td = b.taildep
        out_cases.append({
            "family": "tawn", "rotation": 0, "pars": pars,
            "native": f"Bicop(family=BicopFamily.tawn, parameters=[{', '.join(repr(x) for x in p)}])",
            "pdf": base._nan_to_none(b.pdf(U)),
            "cdf": base._nan_to_none(b.cdf(U)),
            "h1": base._nan_to_none(b.hfunc1(U)),
            "h2": base._nan_to_none(b.hfunc2(U)),
            "hinv1": base._nan_to_none(b.hinv1(U)),
            "hinv2": base._nan_to_none(b.hinv2(U)),
            "tau": float(b.tau),
            "lambda_L": float(td[0, 0]),
            "lambda_U": float(td[1, 1]),
            "A": None,
        })
        errs = base._fd_checks(b, "tawn", U)
        errs["param_cdf_abs_error"] = np.abs(b.cdf(U) - textbook_cdf(pars, U))
        for name, err in errs.items():
            worst[name + "_max"] = max(worst.get(name + "_max", 0.0), float(np.max(err)))
            worst[name + "_median"] = max(worst.get(name + "_median", 0.0), float(np.median(err)))

    version = re.search(r'__version__\s*=\s*"([^"]+)"',
                        (ROOT / "pmcprg" / "__init__.py").read_text()).group(1)
    with open(HERE / "pickands_points.csv", newline="") as fh:
        tgrid = [float(r["t"]) for r in csv.DictReader(fh)]
    doc = {
        "format": 1,
        "about": ("FR-11 interior parity references (families beyond the pilot), generated by "
                  "scripts/parity/gen_pyvinecopulib_extra.py - do not edit by hand. Floats are "
                  "shortest round-trip repr of IEEE-754 doubles; null = not recorded."),
        "reference": {"package": "pyvinecopulib", "version": pv.__version__,
                      "backend": "vinecopulib (C++), bundled"},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform(), "machine": platform.machine()},
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "repository_version": version,
        "command": ".venv-parity/bin/python scripts/parity/gen_pyvinecopulib_extra.py",
        "definitions": {
            "pars": "reference-agnostic key (README): tawn [theta, psi_u, psi_v], psi_u the "
                    "weight of -log u; native = [psi1, psi2, theta] (convention_checks.tawn_weights)",
            "pdf": "c(u, v)", "cdf": "C(u, v)",
            "h1": "hfunc1 = P(U2 <= v | U1 = u) = dC/du",
            "h2": "hfunc2 = P(U1 <= u | U2 = v) = dC/dv",
            "hinv1": "hinv1: the y with h1(u, y) = v (inverse in the second argument)",
            "hinv2": "hinv2: the x with h2(x, v) = u (inverse in the first argument)",
            "tau": "Bicop.tau", "lambda_L": "taildep[0, 0]", "lambda_U": "taildep[1, 1]",
            "A": "not exposed",
        },
        "convention_checks": checks,
        "points": U.tolist(),
        "pickands_points": tgrid,
        "grid": read_grid(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        head = json.dumps(doc, indent=1, ensure_ascii=False)
        fh.write(head[:-2] + ',\n "cases": [\n')
        fh.write(",\n".join("  " + json.dumps(c, ensure_ascii=False) for c in out_cases))
        fh.write("\n ]\n}\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(out_cases)} cases x {len(U)} points; "
          f"tawn weights: {checks['tawn_weights']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
