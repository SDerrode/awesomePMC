#!/usr/bin/env python3
"""Interior parity references from pyvinecopulib (audit FR-11).

Evaluates pdf, cdf, both h-functions, both h-inverses, Kendall's τ and the
two diagonal tail-dependence coefficients of the families of
``scripts/parity/cases.csv`` on the interior points of
``scripts/parity/points.csv``, and writes them, with their provenance, to
``pmcprg/tests/data/parity/pyvinecopulib.json``. The tests
(``pmcprg/tests/test_parity_interior.py``) read that file only.

The script is deliberately independent of ``pmcprg``: it runs in its own
environment (``.venv-parity``, Python 3.13, pyvinecopulib 1.0.0 — see
``scripts/parity/README.md``) and never imports the package it checks.

Every convention the tests rely on — which argument each h-function
conditions on, which argument each inverse solves for, which argument each
rotation reflects — is *measured* here on the reference itself and written
to the file under ``convention_checks``; the script stops if a check does
not come out as documented in the README.

Usage, from the repository root::

    .venv-parity/bin/python scripts/parity/gen_pyvinecopulib.py
"""

from __future__ import annotations

import csv
import datetime
import json
import math
import platform
import re
import sys
from pathlib import Path

import numpy as np
import pyvinecopulib as pv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "pmcprg" / "tests" / "data" / "parity" / "pyvinecopulib.json"

FAMILY = {
    "gaussian": pv.BicopFamily.gaussian,
    "student": pv.BicopFamily.student,
    "clayton": pv.BicopFamily.clayton,
    "gumbel": pv.BicopFamily.gumbel,
    "frank": pv.BicopFamily.frank,
    "joe": pv.BicopFamily.joe,
}
# Families whose reference CDF is not recorded: pmcprg's Student copula has
# no CDF (``CopulaStudent.cdf`` raises), so there is nothing to compare with.
NO_CDF = {"student"}

# Finite-difference step of the convention checks, and the largest median
# error a check may show. Central differences carry an O(δ²) truncation and
# an ε/δ rounding error: measured at most 7e-6 on the grid (Clayton θ = 9),
# while the wrong convention (h1 for h2, a flipped argument) is off by
# O(0.1). The round trips h(hinv) are exact up to the reference's own
# inversion tolerance: measured at most 1.6e-9 (Joe, numerical inversion).
FD_STEP = 1e-5
FD_TOL = 1e-4
ROUNDTRIP_TOL = 1e-8


def read_points() -> np.ndarray:
    with open(HERE / "points.csv", newline="") as fh:
        rows = [(float(r["u"]), float(r["v"])) for r in csv.DictReader(fh)]
    return np.asfortranarray(np.array(rows, dtype=float))


def read_cases() -> list[dict]:
    with open(HERE / "cases.csv", newline="") as fh:
        cases = []
        for r in csv.DictReader(fh):
            pars = [float(r["par1"])] + ([float(r["par2"])] if r["par2"] else [])
            cases.append({"family": r["family"], "rotation": int(r["rotation"]), "pars": pars})
    return cases


def bicop(family: str, rotation: int, pars: list[float]) -> pv.Bicop:
    return pv.Bicop(family=FAMILY[family], rotation=rotation,
                    parameters=np.array([[p] for p in pars], dtype=float))


def _nan_to_none(values) -> list:
    return [None if not math.isfinite(float(x)) else float(x) for x in np.ravel(values)]


def _fd_checks(b: pv.Bicop, family: str, U: np.ndarray) -> dict:
    """Per-point deviations of the reference from its own documented conventions.

    h1 = ∂C/∂u1 and h2 = ∂C/∂u2 (central differences of the CDF);
    pdf = ∂h1/∂u2 = ∂h2/∂u1 (central differences of the h-functions);
    hinv1 inverts h1 in u2 and hinv2 inverts h2 in u1 (round trips).

    A convention is judged on the median over the points, as in gen_r.R: a
    wrong convention is off by O(0.1) almost everywhere, while a reference's
    local inaccuracy — which the parity tests measure — shows at one or two
    points. The maximum is recorded as well.
    """
    d = FD_STEP
    e1 = np.array([d, 0.0])
    e2 = np.array([0.0, d])
    out = {}
    if family not in NO_CDF:
        out["h1_minus_dC_du1"] = np.abs(b.hfunc1(U) - (b.cdf(U + e1) - b.cdf(U - e1)) / (2 * d))
        out["h2_minus_dC_du2"] = np.abs(b.hfunc2(U) - (b.cdf(U + e2) - b.cdf(U - e2)) / (2 * d))
    pdf = b.pdf(U)
    scale = np.maximum(pdf, 1.0)          # relative above 1, absolute below
    out["pdf_minus_dh1_du2"] = np.abs(
        pdf - (b.hfunc1(U + e2) - b.hfunc1(U - e2)) / (2 * d)) / scale
    out["pdf_minus_dh2_du1"] = np.abs(
        pdf - (b.hfunc2(U + e1) - b.hfunc2(U - e1)) / (2 * d)) / scale
    y = b.hinv1(U)
    out["h1_of_hinv1_minus_level"] = np.abs(
        b.hfunc1(np.asfortranarray(np.column_stack([U[:, 0], y]))) - U[:, 1])
    x = b.hinv2(U)
    out["h2_of_hinv2_minus_level"] = np.abs(
        b.hfunc2(np.asfortranarray(np.column_stack([x, U[:, 1]]))) - U[:, 0])
    for key, err in out.items():
        tol = ROUNDTRIP_TOL if "hinv" in key else FD_TOL
        if not np.median(err) < tol:
            sys.exit(f"convention check failed for {family}: median {key} = {np.median(err):.3g}")
    return out


_REFLECTIONS = {
    "c0(1-u, v)": lambda U: np.column_stack([1.0 - U[:, 0], U[:, 1]]),
    "c0(u, 1-v)": lambda U: np.column_stack([U[:, 0], 1.0 - U[:, 1]]),
    "c0(1-u, 1-v)": lambda U: np.column_stack([1.0 - U[:, 0], 1.0 - U[:, 1]]),
}


def _rotation_check(family: str, rotation: int, pars: list[float], U: np.ndarray) -> dict:
    """Which reflection of the unrotated density the rotated density equals.

    Judged on the median relative error, with the same threshold as gen_r.R
    (see there: copula's rotated densities are off by up to 1.5e-3 at one
    point); the wrong reflections are off by O(1) at most points.
    """
    base = bicop(family, 0, pars)
    rot = bicop(family, rotation, pars)
    pdf = rot.pdf(U)
    rel = {name: np.abs(base.pdf(np.asfortranarray(f(U))) - pdf) / pdf
           for name, f in _REFLECTIONS.items()}
    match = [name for name, e in rel.items() if np.median(e) < 1e-9]
    if len(match) != 1:
        sys.exit(f"rotation {rotation} of {family}: no unique reflection matches "
                 f"({ {k: float(np.median(e)) for k, e in rel.items()} })")
    return {"density_equals": match[0], "rel_error_max": float(np.max(rel[match[0]])),
            "rel_error_median": float(np.median(rel[match[0]])),
            "tau_rotated_over_tau_base": rot.tau / base.tau}


def main() -> int:
    U = read_points()
    cases = read_cases()
    out_cases = []
    checks = {"fd_step": FD_STEP, "fd_tolerance": FD_TOL, "roundtrip_tolerance": ROUNDTRIP_TOL,
              "judged_on": "median over the points (max recorded); "
                           "pdf errors relative above pdf = 1, absolute below",
              "per_family": {}, "rotations": {}}
    for case in cases:
        fam, rot, pars = case["family"], case["rotation"], case["pars"]
        b = bicop(fam, rot, pars)
        td = b.taildep
        entry = {
            "family": fam, "rotation": rot, "pars": pars,
            "native": f"Bicop(family=BicopFamily.{fam}, rotation={rot}, "
                      f"parameters=[{', '.join(repr(p) for p in pars)}])",
            "pdf": _nan_to_none(b.pdf(U)),
            "cdf": None if fam in NO_CDF else _nan_to_none(b.cdf(U)),
            "h1": _nan_to_none(b.hfunc1(U)),
            "h2": _nan_to_none(b.hfunc2(U)),
            "hinv1": _nan_to_none(b.hinv1(U)),
            "hinv2": _nan_to_none(b.hinv2(U)),
            "tau": float(b.tau),
            "lambda_L": float(td[0, 0]),
            "lambda_U": float(td[1, 1]),
        }
        out_cases.append(entry)
        key = f"{fam}/{rot}"
        worst = checks["per_family"].setdefault(key, {})
        for name, err in _fd_checks(b, fam, U).items():
            worst[name + "_max"] = max(worst.get(name + "_max", 0.0), float(np.max(err)))
            worst[name + "_median"] = max(worst.get(name + "_median", 0.0), float(np.median(err)))
        if rot != 0:
            rc = _rotation_check(fam, rot, pars, U)
            prev = checks["rotations"].get(key)
            if prev is not None and prev["density_equals"] != rc["density_equals"]:
                sys.exit(f"{key}: the reflection depends on the parameter")
            if prev is None or rc["rel_error_max"] > prev["rel_error_max"]:
                checks["rotations"][key] = rc

    version = re.search(r'__version__\s*=\s*"([^"]+)"',
                        (ROOT / "pmcprg" / "__init__.py").read_text()).group(1)
    doc = {
        "format": 1,
        "about": ("FR-11 interior parity references, generated by "
                  "scripts/parity/gen_pyvinecopulib.py - do not edit by hand. "
                  "Floats are shortest round-trip repr of IEEE-754 doubles; null = not recorded."),
        "reference": {"package": "pyvinecopulib", "version": pv.__version__,
                      "backend": "vinecopulib (C++), bundled"},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform(), "machine": platform.machine()},
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "repository_version": version,
        "command": ".venv-parity/bin/python scripts/parity/gen_pyvinecopulib.py",
        "definitions": {
            "pars": "reference-native parameters: gaussian [rho]; student [rho, nu]; "
                    "clayton/gumbel/frank/joe [theta] of the unrotated family (theta > 0 "
                    "for every rotation); rotation in degrees",
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
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        head = json.dumps(doc, indent=1, ensure_ascii=False)
        fh.write(head[:-2] + ',\n "cases": [\n')
        fh.write(",\n".join("  " + json.dumps(c, ensure_ascii=False) for c in out_cases))
        fh.write("\n ]\n}\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(out_cases)} cases x {len(U)} points")
    for key, rc in checks["rotations"].items():
        print(f"  {key}: density = {rc['density_equals']}, tau ratio {rc['tau_rotated_over_tau_base']:+.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
