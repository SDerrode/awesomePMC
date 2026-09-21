#!/usr/bin/env python3
"""Interior parity references from pyvinecopulib for Joe's BB1, BB6, BB7, BB8 (audit FR-11, wave 2).

Evaluates pdf, cdf, both h-functions, both h-inverses, Kendall's τ and the
two diagonal tail-dependence coefficients of the cases of
``scripts/parity/cases_bb.csv`` — BB1 in its four rotations, BB6, BB7 and
BB8 unrotated (pmcprg registers no rotation of the last three) — on the
interior points of ``scripts/parity/points.csv``, and writes them, with
their provenance, to ``pmcprg/tests/data/parity/bb_pyvinecopulib.json``,
read by ``pmcprg/tests/test_parity_bb.py``.

Which native parameter is which is *measured*, not assumed: the reference's
CDF is compared with the textbook formula of the reference-agnostic key
(``[theta, delta]``, Joe 1997, ch. 5 — ``textbook_cdf`` below, the formulas
of ``scripts/parity/README.md``) under both orders of the two parameters,
and the order that matches is recorded under
``convention_checks.parameter_order``; the rotations and the h-function
conventions are measured as in ``gen_pyvinecopulib.py``, whose helpers this
script reuses. It never imports ``pmcprg``.

Usage, from the repository root::

    .venv-parity/bin/python scripts/parity/gen_pyvinecopulib_bb.py
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
OUT = ROOT / "pmcprg" / "tests" / "data" / "parity" / "bb_pyvinecopulib.json"

# The pilot's helpers (bicop, _rotation_check) look families up here.
base.FAMILY.update(bb1=pv.BicopFamily.bb1, bb6=pv.BicopFamily.bb6,
                   bb7=pv.BicopFamily.bb7, bb8=pv.BicopFamily.bb8)

# As gen_r_bb.R: the swapped order, where the reference accepts it at all,
# is off by O(0.01) at most points (BB6 (1.8, 1.4) against (1.4, 1.8): 1e-2).
PARAM_TOL = 1e-9


def read_grid() -> list[list]:
    """Every case of ``cases_bb.csv``, as ``[family, rotation, [theta, delta]]`` (the tests read it)."""
    with open(HERE / "cases_bb.csv", newline="") as fh:
        return [[r["family"], int(r["rotation"]), [float(r["par1"]), float(r["par2"])]]
                for r in csv.DictReader(fh)]


def textbook_base(family: str, th: float, de: float):
    """The unrotated textbook CDF of the key (Joe 1997, ch. 5; README)."""
    if family == "bb1":
        return lambda u, v: (1 + ((u ** -th - 1) ** de + (v ** -th - 1) ** de) ** (1 / de)) ** (-1 / th)
    if family == "bb6":
        def C(u, v):
            p, q = -np.log(1 - (1 - u) ** th), -np.log(1 - (1 - v) ** th)
            return 1 - (1 - np.exp(-(p ** de + q ** de) ** (1 / de))) ** (1 / th)
        return C
    if family == "bb7":
        def C(u, v):
            a, b = 1 - (1 - u) ** th, 1 - (1 - v) ** th
            return 1 - (1 - (a ** -de + b ** -de - 1) ** (-1 / de)) ** (1 / th)
        return C
    if family == "bb8":
        def C(u, v):
            eta = 1 - (1 - de) ** th
            A = (1 - (1 - de * u) ** th) * (1 - (1 - de * v) ** th) / eta
            return (1 - (1 - A) ** (1 / th)) / de
        return C
    raise KeyError(family)


def textbook_cdf(family: str, rotation: int, pars: list[float], U: np.ndarray) -> np.ndarray:
    C = textbook_base(family, *pars)
    u, v = U[:, 0], U[:, 1]
    return {0: lambda: C(u, v), 90: lambda: v - C(1 - u, v), 270: lambda: u - C(u, 1 - v),
            180: lambda: u + v - 1 + C(1 - u, 1 - v)}[rotation]()


def native(family: str, rotation: int, pars: list[float], U: np.ndarray) -> tuple[list[float], str]:
    """The native parameters whose CDF is the case's textbook CDF — each order tried."""
    tb = textbook_cdf(family, rotation, pars, U)
    hits = []
    for order, label in ((pars, "[theta, delta]"), (pars[::-1], "[delta, theta]")):
        try:
            b = base.bicop(family, rotation, list(order))
        except Exception:         # out of the reference's bounds: not this order
            continue
        if float(np.median(np.abs(b.cdf(U) - tb))) < PARAM_TOL:
            hits.append((list(order), label))
    if len(hits) != 1:
        sys.exit(f"{family}/{rotation} {pars}: no unique parameter order matches")
    return hits[0]


def main() -> int:
    U = base.read_points()
    grid = read_grid()
    out_cases = []
    checks = {"fd_step": base.FD_STEP, "fd_tolerance": base.FD_TOL,
              "roundtrip_tolerance": base.ROUNDTRIP_TOL, "param_tolerance": PARAM_TOL,
              "judged_on": "median over the points (max recorded); param_* absolute against "
                           "the textbook CDF; pdf errors relative above pdf = 1, absolute below",
              "per_family": {}, "rotations": {}, "parameter_order": {}}
    for fam, rot, pars in grid:
        p, label = native(fam, rot, pars, U)
        prev = checks["parameter_order"].setdefault(fam, label)
        if prev != label:
            sys.exit(f"{fam}: the parameter order depends on the case")
        b = base.bicop(fam, rot, p)
        td = b.taildep
        out_cases.append({
            "family": fam, "rotation": rot, "pars": pars,
            "native": f"Bicop(family=BicopFamily.{fam}, rotation={rot}, "
                      f"parameters=[{', '.join(repr(x) for x in p)}])",
            "pdf": base._nan_to_none(b.pdf(U)),
            "cdf": base._nan_to_none(b.cdf(U)),
            "h1": base._nan_to_none(b.hfunc1(U)),
            "h2": base._nan_to_none(b.hfunc2(U)),
            "hinv1": base._nan_to_none(b.hinv1(U)),
            "hinv2": base._nan_to_none(b.hinv2(U)),
            "tau": float(b.tau),
            "lambda_L": float(td[0, 0]),
            "lambda_U": float(td[1, 1]),
        })
        key = f"{fam}/{rot}"
        worst = checks["per_family"].setdefault(key, {})
        errs = base._fd_checks(b, fam, U)
        errs["param_cdf_abs_error"] = np.abs(b.cdf(U) - textbook_cdf(fam, rot, pars, U))
        for name, err in errs.items():
            worst[name + "_max"] = max(worst.get(name + "_max", 0.0), float(np.max(err)))
            worst[name + "_median"] = max(worst.get(name + "_median", 0.0), float(np.median(err)))
        if rot != 0:
            rc = base._rotation_check(fam, rot, p, U)
            prev = checks["rotations"].get(key)
            if prev is not None and prev["density_equals"] != rc["density_equals"]:
                sys.exit(f"{key}: the reflection depends on the parameter")
            if prev is None or rc["rel_error_max"] > prev["rel_error_max"]:
                checks["rotations"][key] = rc

    version = re.search(r'__version__\s*=\s*"([^"]+)"',
                        (ROOT / "pmcprg" / "__init__.py").read_text()).group(1)
    doc = {
        "format": 1,
        "about": ("FR-11 interior parity references (BB1, BB6, BB7, BB8), generated by "
                  "scripts/parity/gen_pyvinecopulib_bb.py - do not edit by hand. Floats are "
                  "shortest round-trip repr of IEEE-754 doubles; null = not recorded."),
        "reference": {"package": "pyvinecopulib", "version": pv.__version__,
                      "backend": "vinecopulib (C++), bundled"},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform(), "machine": platform.machine()},
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "repository_version": version,
        "command": ".venv-parity/bin/python scripts/parity/gen_pyvinecopulib_bb.py",
        "definitions": {
            "pars": "reference-agnostic key (README): [theta, delta] of the unrotated family "
                    "(Joe 1997, ch. 5), positive for every rotation; native = the Bicop call "
                    "(convention_checks.parameter_order)",
            "pdf": "c(u, v)", "cdf": "C(u, v)",
            "h1": "hfunc1 = P(U2 <= v | U1 = u) = dC/du",
            "h2": "hfunc2 = P(U1 <= u | U2 = v) = dC/dv",
            "hinv1": "hinv1: the y with h1(u, y) = v (inverse in the second argument)",
            "hinv2": "hinv2: the x with h2(x, v) = u (inverse in the first argument)",
            "tau": "Bicop.tau (sign flipped by the reference for 90/270)",
            "lambda_L": "taildep[0, 0]", "lambda_U": "taildep[1, 1]",
        },
        "convention_checks": checks,
        "points": U.tolist(),
        "grid": grid,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        head = json.dumps(doc, indent=1, ensure_ascii=False)
        fh.write(head[:-2] + ',\n "cases": [\n')
        fh.write(",\n".join("  " + json.dumps(c, ensure_ascii=False) for c in out_cases))
        fh.write("\n ]\n}\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(out_cases)} cases x {len(U)} points")
    for fam, label in checks["parameter_order"].items():
        print(f"  {fam}: parameters = {label}")
    for key, rc in checks["rotations"].items():
        print(f"  {key}: density = {rc['density_equals']}, tau ratio {rc['tau_rotated_over_tau_base']:+.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
