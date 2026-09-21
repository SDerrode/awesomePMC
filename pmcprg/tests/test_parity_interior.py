"""Interior parity with pyvinecopulib, VineCopula and R ``copula`` — audit FR-11 (pilot).

pdf, cdf, both h-functions, both h-inverses, Kendall's τ(θ), λ_L and λ_U of
Gaussian, Student, Frank, Clayton, Gumbel (GH) and Joe — with the survival
(180°) and 90°/270° rotations of the last three — against three independent
implementations, on 60 parameter sets × 25 interior points of [0.01, 0.99]²
(``scripts/parity/cases.csv``, ``points.csv``). The tails are FR-1's
(``test_copula_limits.py``): the references clamp or cap there.

The reference values are tables generated **offline** by
``scripts/parity/gen_pyvinecopulib.py`` (pyvinecopulib 1.0.0) and
``scripts/parity/gen_r.R`` (VineCopula 2.6.1, copula 1.1-7), versioned in
``data/parity/`` with their provenance; nothing here needs R or
pyvinecopulib. ``mpmath`` (a dev dependency) arbitrates the discrepancies.

Convention mapping
------------------
Every line below was measured on the references by the generators (their
``convention_checks``, re-read by :func:`test_recorded_conventions`) and on
pmcprg by :func:`test_pmcprg_rotation_and_transpose_conventions` — see
``scripts/parity/README.md`` for the full table.

* **Parameters.** pmcprg takes Kendall's τ; the references ρ (Gauss, t, with
  ν = ``df``) or θ. pmcprg is built at τ(θ) from its own maps —
  (2/π) asin ρ, ``kendall_tau_frank``, θ/(θ+2), 1 − 1/θ, the Joe
  digamma/series form — and its constructor returns θ to 1.0e-14 (Frank,
  Brent's ``xtol``) or 5.1e-16 (the others). The key θ of a table is the θ of
  the *unrotated* family, positive for every rotation: pyvinecopulib takes it
  with ``rotation=90``, VineCopula as family 23/24/26 or 33/34/36 with
  **−θ**, copula as ``rotCopula(base(θ), flip)``; pmcprg's 90°/270° families
  take τ = −τ(θ).
* **Rotations.** Everywhere, 90° reflects the first argument and 270° the
  second: c₉₀(u, v) = c₀(1 − u, v), c₂₇₀(u, v) = c₀(u, 1 − v),
  c₁₈₀(u, v) = c₀(1 − u, 1 − v); in copula ``flip = c(TRUE, FALSE)``,
  ``c(FALSE, TRUE)``, ``c(TRUE, TRUE)``. pmcprg's 180° families are the
  ``SURVIVAL_*`` members.
* **h-functions.** h1(u, v) = ∂C/∂u = P(V ≤ v | U = u) is pmcprg's
  ``conditional_cdf(v, u)`` (conditioning variable second); hinv1 inverts it
  in its second argument — pmcprg's ``inv_h(w, u)``. h2(u, v) = ∂C/∂v is the
  ``conditional_cdf(u, v)`` of the **transposed** copula
  (``CopulaVirt.transposed()``), and hinv2(w, v) its ``inv_h(w, v)`` — the
  copula itself for every exchangeable family, 270° for 90° and 90° for 270°
  (c₉₀(u, v) = c₂₇₀(v, u) for an exchangeable base). ``BivariateLaw``
  conditions on its right margin through it
  (:func:`test_bivariate_law_conditions_on_either_margin`).
* **VineCopula clamps h** to [10⁻¹², 1 − 10⁻¹²] (it returns exactly 1e-12
  where the true value is 1.1e-20); pmcprg is compared clamped the same way.
* **Missing reference values** (``null`` in a table): Student's CDF (pmcprg
  has none; copula refuses a non-integer ν); copula's h-values and λ for the
  rotated families (``README``: its inverse is not implemented for
  ``rotCopula`` and its forward ``cCopula`` omits the ``1 −`` of a flipped
  second margin — 0.178 for ∂C/∂u = 0.822).

Tolerances
----------
Each tolerance of :data:`TOL` is a small multiple of the largest error
measured between pmcprg and a reference over the whole grid, stated next to
it. vinecopulib's own parity tolerances (10⁻⁴ for pdf/cdf/h/h⁻¹, 10⁻² for τ
and λ) are bounds, not targets: pmcprg agrees to 10⁻¹³ or better wherever
the reference itself is accurate. Where it does not, the disagreement is in
one of two registers, each entry proven by ``mpmath`` at its worst point:

* :data:`PMCPRG_LIMITED` — the three references agree and pmcprg does not:
  the Gaussian CDF, 1.2e-14 (7e-11 relative) from mpmath where the
  references are within 2.2e-16 — the accuracy documented in
  ``pmcprg/copulas/elliptical/gaussian.py`` (composite 8-point
  Gauss–Legendre, relative error below 2·10⁻¹⁰), not a defect;
* :data:`REFERENCE_LIMITED` — a reference is off and pmcprg agrees with
  mpmath: vinecopulib's Frank CDF/h near (1, 1) and the numerical h-inverses
  of vinecopulib and VineCopula (Frank, Gumbel, Joe; the two give *identical*
  values for Gumbel and Joe), VineCopula's Frank τ (linear interpolation in a
  table), copula's Frank h for θ = −30 and its symbolic densities of the
  rotated Joe copula.

The references' mutual agreement outside these entries is asserted by
:func:`test_references_agree_with_each_other`.

One pmcprg defect surfaced through h2 and is fixed: ``BivariateLaw``
conditioned on its right margin by swapping the arguments of the copula's
h-function and density — right for an exchangeable copula only, so for the
90°/270° rotations and the Tawn models it returned the conditional law of
the other side (Clayton 90°, τ = −0.5: P(U ≤ 0.3 | V = 0.8) read 0.601 for
0.535). It now goes through the transposed copula.
"""
from __future__ import annotations

import functools
import json
import math
from dataclasses import dataclass
from pathlib import Path

try:                              # a dev dependency: the minimum-versions CI job
    import mpmath as mp           # installs the runtime ones only, and the table
except ImportError:               # comparisons must still run there
    mp = None
import numpy as np
import pytest
import scipy

from pmcprg.copulas import CopulaEnum
from pmcprg.copulas.archimedean.frank import kendall_tau_frank
from pmcprg.copulas.archimedean.joe import _joe_tau_from_theta

DATA = Path(__file__).resolve().parent / "data" / "parity"
REFERENCES = ("pyvinecopulib", "vinecopula", "rcopula")
VECTOR_QUANTITIES = ("pdf", "cdf", "h1", "h2", "hinv1", "hinv2")
ROTATIONS = (0, 180, 90, 270)


@functools.cache
def _table(name: str) -> dict:
    return json.loads((DATA / f"{name}.json").read_text())


@functools.cache
def _cases(name: str) -> dict:
    return {(c["family"], c["rotation"], tuple(c["pars"])): c for c in _table(name)["cases"]}


POINTS = np.array(_table("pyvinecopulib")["points"], dtype=float)
CASE_KEYS = tuple(_cases("pyvinecopulib"))
FAMILY_ROTATIONS = tuple(dict.fromkeys((f, r) for f, r, _ in CASE_KEYS))

# (family, rotation) of a table → (pmcprg registry member, its transpose).
PMCPRG = {
    ("gaussian", 0): ("GAUSSIAN", "GAUSSIAN"),
    ("student", 0): ("STUDENT", "STUDENT"),
    ("frank", 0): ("FRANK", "FRANK"),
    ("clayton", 0): ("CLAYTON", "CLAYTON"),
    ("clayton", 180): ("SURVIVAL_CLAYTON", "SURVIVAL_CLAYTON"),
    ("clayton", 90): ("CLAYTON90", "CLAYTON270"),
    ("clayton", 270): ("CLAYTON270", "CLAYTON90"),
    ("gumbel", 0): ("GH", "GH"),
    ("gumbel", 180): ("SURVIVAL_GH", "SURVIVAL_GH"),
    ("gumbel", 90): ("GH90", "GH270"),
    ("gumbel", 270): ("GH270", "GH90"),
    ("joe", 0): ("JOE", "JOE"),
    ("joe", 180): ("SURVIVAL_JOE", "SURVIVAL_JOE"),
    ("joe", 90): ("JOE90", "JOE270"),
    ("joe", 270): ("JOE270", "JOE90"),
}

# VineCopula returns its h-functions clamped to this interval.
VINECOPULA_H_RANGE = (1e-12, 1.0 - 1e-12)


# --------------------------------------------------------------------------
# pmcprg side
# --------------------------------------------------------------------------

def _tau_of_theta(family: str, pars: tuple) -> float:
    """pmcprg's τ(θ) of the unrotated family (module docstring)."""
    th = pars[0]
    if family in ("gaussian", "student"):
        return 2.0 / math.pi * math.asin(th)
    if family == "frank":
        return kendall_tau_frank(th)
    if family == "clayton":
        return th / (th + 2.0)
    if family == "gumbel":
        return 1.0 - 1.0 / th
    if family == "joe":
        return _joe_tau_from_theta(th)
    raise KeyError(family)


def _build(family: str, rotation: int, pars: tuple, tau: float | None = None):
    """(copula, its ``transposed()``) of pmcprg for a table case."""
    member, transpose = PMCPRG[(family, rotation)]
    if tau is None:
        tau = _tau_of_theta(family, pars) * (-1.0 if rotation in (90, 270) else 1.0)
    kwargs = {"tau_k": tau}
    if family == "student":
        kwargs["df"] = pars[1]
    cop = getattr(CopulaEnum, member).klass(**kwargs)
    tr = cop.transposed()
    assert type(tr) is getattr(CopulaEnum, transpose).klass and tr.params == cop.params
    return cop, tr


def _theta(cop) -> float:
    """θ (ρ for the elliptical families) of pmcprg's copula or of its rotated base."""
    return getattr(cop, "_base", cop).theta


@functools.cache
def _ours(family: str, rotation: int, pars: tuple) -> dict:
    cop, tr = _build(family, rotation, pars)
    u, v = POINTS[:, 0].copy(), POINTS[:, 1].copy()
    lam_l, lam_u = cop.tail_dependence()
    return {
        "theta": _theta(cop),
        "pdf": cop.pdf_array(POINTS),
        "cdf": None if family == "student" else cop.cdf_array(POINTS),
        "h1": np.array([cop.conditional_cdf(b, a) for a, b in POINTS]),
        "h2": np.array([tr.conditional_cdf(a, b) for a, b in POINTS]),
        "hinv1": cop.inv_h_array(v, u),
        "hinv2": tr.inv_h_array(u, v),
        "tau": float(cop.params["tau_k"]),
        "lambda_L": lam_l,
        "lambda_U": lam_u,
    }


def _ref_vector(case: dict, q: str) -> np.ndarray | None:
    values = case[q]
    if values is None:
        return None
    assert None not in values, f"partial null in {case['native']} {q}"
    return np.array(values, dtype=float)


def _errors(q: str, x: np.ndarray, ref: np.ndarray, reference: str) -> np.ndarray:
    """|x − ref|, relative for the density; VineCopula's h compared clamped."""
    if reference == "vinecopula" and q in ("h1", "h2"):
        x = np.clip(x, *VINECOPULA_H_RANGE)
    d = np.abs(np.asarray(x, dtype=float) - ref)
    return d / np.abs(ref) if q == "pdf" else d


# --------------------------------------------------------------------------
# Tolerances (module docstring). Absolute, the density's relative.
# --------------------------------------------------------------------------

TOL = {
    # 8.5e-14: Gaussian ρ = −0.9 at (0.01, 0.01), c = 1.6e-21, against
    # VineCopula and copula, themselves 6.5e-14 from mpmath there (pmcprg
    # ≤ 2.9e-14 from mpmath on the whole grid).
    "pdf": 2e-13,
    # 1.9e-15: Clayton θ = 9, the same against all three references.
    "cdf": 5e-15,
    # 4.2e-14: copula, Frank θ = −8 (its own error: pmcprg is 2.1e-15 from
    # mpmath); 1.9e-14 against pyvinecopulib (Gumbel), 4.4e-15 against
    # VineCopula (Student).
    "h1": 1e-13,
    "h2": 1e-13,
    # 2.2e-14: copula, Gumbel θ = 3 — uniroot() to tol 1e-15 (pmcprg 2.2e-16
    # from mpmath); 5.7e-15 against the two vine libraries (rotated Clayton).
    "hinv1": 5e-14,
    "hinv2": 5e-14,
    # 3.8e-15: Frank against pyvinecopulib, Joe against copula.
    "tau": 1e-14,
    # 2.5e-16: Student against VineCopula and copula.
    "lambda_L": 1e-15,
    "lambda_U": 1e-15,
}

# The three references agree and pmcprg does not (module docstring).
PMCPRG_LIMITED = {
    # 1.2e-14 against each reference (ρ = −0.9 at (0.2, 0.35), C = 1.6e-4:
    # 7.1e-11 relative, within the ≤ 2e-10 of gaussian.py's docstring); the
    # references agree with one another to 1.1e-16. See
    # test_gaussian_cdf_gap_is_pmcprg_quadrature.
    ("gaussian", "cdf"): 3e-14,
}


@dataclass(frozen=True)
class Limited:
    """A cell where the *reference* is off — its tolerance is the reference's error.

    ``pars`` None: every parameter set of the family. Each entry is proven
    by :func:`test_reference_limited_entries_are_the_reference` with mpmath
    at its worst point, and must still be needed (worst error above
    :data:`TOL`).
    """

    reference: str
    family: str
    rotations: tuple
    pars: tuple | None
    quantities: tuple
    tol: float
    why: str

    def covers(self, reference: str, family: str, rotation: int, pars: tuple, q: str) -> bool:
        return (reference == self.reference and family == self.family
                and rotation in self.rotations and q in self.quantities
                and (self.pars is None or tuple(pars) in self.pars))


HINV = ("hinv1", "hinv2")
REFERENCE_LIMITED = (
    Limited("pyvinecopulib", "frank", (0,), ((12.0,),), ("cdf",), 3e-12,
            "vinecopulib's Frank CDF near (1, 1): 1.4e-12 at (0.99, 0.99)"),
    Limited("pyvinecopulib", "frank", (0,), ((12.0,),), ("h1", "h2"), 3e-11,
            "vinecopulib's Frank h near (1, 1): 1.5e-11 at (0.99, 0.99)"),
    Limited("pyvinecopulib", "frank", (0,), None, HINV, 6e-11,
            "numerical inversion: 2.9e-11, e.g. at (0.5, 0.5), where hinv = 0.5 exactly"),
    Limited("pyvinecopulib", "gumbel", ROTATIONS, None, HINV, 3e-11,
            "numerical inversion: 1.1e-11 (θ = 7, (0.9, 0.9))"),
    Limited("pyvinecopulib", "joe", ROTATIONS, None, HINV, 3e-10,
            "numerical inversion: 1.2e-10 (θ = 2.5, rotated)"),
    Limited("vinecopula", "frank", (0,), None, HINV, 1e-11,
            "numerical inversion: 4.5e-12 (θ = −8, (0.01, 0.01))"),
    Limited("vinecopula", "gumbel", ROTATIONS, None, HINV, 3e-11,
            "numerical inversion, bit for bit vinecopulib's: 1.1e-11"),
    Limited("vinecopula", "joe", ROTATIONS, None, HINV, 3e-10,
            "numerical inversion, bit for bit vinecopulib's: 1.2e-10"),
    Limited("vinecopula", "frank", (0,), None, ("tau",), 1e-3,
            "VineCopula:::frankTau is approx(frankParGrid, frankTauVals), a linear "
            "interpolation: 5.5e-4 at θ = 2"),
    Limited("rcopula", "frank", (0,), ((-30.0,),), ("h1", "h2"), 2e-5,
            "copula's Frank h for θ < 0 (rosenblatt() through absdPsi(log = FALSE)): "
            "8.8e-6 at θ = −30, (0.9, 0.1)"),
    Limited("rcopula", "frank", (0,), ((-30.0,),), HINV, 5e-5,
            "uniroot() on that h: 2.6e-5"),
    Limited("rcopula", "joe", (180, 90, 270), ((2.5,),), ("pdf",), 2e-11,
            "rotExplicitCopula evaluates its symbolic density (.ExplicitCopula.algr, "
            "'pdfalgr'), not the base's: 7.9e-12 relative"),
    Limited("rcopula", "joe", (180, 90, 270), ((7.0,),), ("pdf",), 3e-3,
            "the same symbolic density: 1.5e-3 relative at θ = 7"),
    Limited("rcopula", "joe", (180, 90, 270), ((2.5,),), ("cdf",), 5e-14,
            "the symbolic CDF ('cdfalgr'): 1.8e-14"),
    Limited("rcopula", "joe", (180, 90, 270), ((7.0,),), ("cdf",), 3e-6,
            "the symbolic CDF: 1.1e-6 at θ = 7"),
)


def _limited(reference: str, family: str, rotation: int, pars: tuple, q: str) -> Limited | None:
    hits = [e for e in REFERENCE_LIMITED if e.covers(reference, family, rotation, pars, q)]
    assert len(hits) <= 1, hits
    return hits[0] if hits else None


# The Student copula evaluates scipy's t distribution (its quantile maps u to
# the t scale), whose accuracy changed with scipy: t.ppf is 2.3e-9 relative
# from mpmath up to scipy 1.12 and 1.6e-11 in 1.13-1.16, and the table
# comparisons above hold from 1.17 on. Maxima of pmcprg against the three
# references on the whole Student grid, measured under scipy 1.10.1 / 1.12.0
# and 1.13.1 / 1.16.2 (same values within each tier), λ and τ unaffected:
#   < 1.13:        pdf 2.83e-8 (relative), h 1.03e-9, h-inverses 1.04e-9;
#   1.13 to 1.16:  pdf 3.10e-11, h 4.12e-12, h-inverses 3.90e-12.
# The minimum-versions CI job runs scipy 1.10, the lower bound in pyproject.
_SCIPY = tuple(int(x) for x in scipy.__version__.split(".")[:2])
STUDENT_TOL_OLD_SCIPY = (
    {"pdf": 1e-7, "h1": 3e-9, "h2": 3e-9, "hinv1": 3e-9, "hinv2": 3e-9} if _SCIPY < (1, 13) else
    {"pdf": 1e-10, "h1": 1.5e-11, "h2": 1.5e-11, "hinv1": 1.5e-11, "hinv2": 1.5e-11}
    if _SCIPY < (1, 17) else {}
)


def _tolerance(reference: str, family: str, rotation: int, pars: tuple, q: str) -> float:
    entry = _limited(reference, family, rotation, pars, q)
    if entry is not None:
        return entry.tol
    tol = PMCPRG_LIMITED.get((family, q), TOL[q])
    if family == "student":
        tol = max(tol, STUDENT_TOL_OLD_SCIPY.get(q, 0.0))
    return tol


# --------------------------------------------------------------------------
# The tables
# --------------------------------------------------------------------------

def test_tables_share_grid_and_carry_provenance():
    grid = np.loadtxt(Path(__file__).resolve().parents[2] / "scripts" / "parity" / "points.csv",
                      delimiter=",", skiprows=1)
    np.testing.assert_array_equal(POINTS, grid)
    assert POINTS.min() >= 0.01 and POINTS.max() <= 0.99          # interior only
    for name in REFERENCES:
        t = _table(name)
        np.testing.assert_array_equal(np.array(t["points"], dtype=float), POINTS)
        assert tuple(_cases(name)) == CASE_KEYS
        for key in ("reference", "environment", "generated", "repository_version",
                    "command", "definitions", "convention_checks"):
            assert t[key], (name, key)
        assert t["reference"]["version"]
    assert set(FAMILY_ROTATIONS) == set(PMCPRG)


EXPECTED_REFLECTION = {180: "c0(1-u, 1-v)", 90: "c0(1-u, v)", 270: "c0(u, 1-v)"}


@pytest.mark.parametrize("name", REFERENCES)
def test_recorded_conventions(name):
    """The conventions of the module docstring are the ones the generators measured."""
    checks = _table(name)["convention_checks"]
    for key, rc in checks["rotations"].items():
        rotation = int(key.split("/")[1])
        assert rc["density_equals"] == EXPECTED_REFLECTION[rotation], (name, key)
        assert rc["tau_rotated_over_tau_base"] == pytest.approx(-1.0 if rotation in (90, 270) else 1.0)
    assert {k for k in checks["rotations"]} == {f"{f}/{r}" for f, r in FAMILY_ROTATIONS if r}
    for key, fam_checks in checks["per_family"].items():
        for check, value in fam_checks.items():
            if check.endswith("_median"):
                tol = checks["roundtrip_tolerance"] if "hinv" in check else checks["fd_tolerance"]
                assert value < tol, (name, key, check, value)


@pytest.mark.parametrize("family", ["clayton", "gumbel", "joe"])
def test_pmcprg_rotation_and_transpose_conventions(family):
    """pmcprg's own conventions, measured like the references' (module docstring)."""
    theta = (2.5,)
    base, _ = _build(family, 0, theta)
    u, v = POINTS[:, 0], POINTS[:, 1]
    ref = {180: np.column_stack([1 - u, 1 - v]), 90: np.column_stack([1 - u, v]),
           270: np.column_stack([u, 1 - v])}
    for rotation, reflected in ref.items():
        cop, tr = _build(family, rotation, theta)
        # pmcprg never forms the reflected argument 1 − u (log1p kernels),
        # the right-hand side here does: a few ulps, measured 2.7e-15
        # (Joe 180°; 1.4e-14 at θ = 7, off this test).
        np.testing.assert_allclose(cop.pdf_array(POINTS), base.pdf_array(reflected), rtol=1e-14)
        # The transpose used for h2 has the transposed density: bit for bit
        # for Clayton and Joe, 5.4e-16 for Gumbel (its kernel is not
        # evaluated symmetrically in its two coordinates).
        np.testing.assert_allclose(tr.pdf_array(POINTS[:, ::-1]), cop.pdf_array(POINTS), rtol=2e-15)
        assert np.sign(cop.params["tau_k"]) == (-1.0 if rotation in (90, 270) else 1.0)


@pytest.mark.parametrize("family, rotation", FAMILY_ROTATIONS,
                         ids=[f"{f}{r}" for f, r in FAMILY_ROTATIONS])
def test_pmcprg_theta_round_trip(family, rotation):
    """pmcprg's constructor returns θ at pmcprg's τ(θ) — the evaluations are at θ."""
    # Measured: 1.0e-14 for Frank (brentq xtol = 1e-14 in find_theta_frank)
    # with scipy >= 1.17, 3.03e-14 with scipy 1.10-1.16 (θ = 0.2, τ = 0.022:
    # the Debye integral's last bits); 5.1e-16 for every other family.
    tol = 1e-13 if family == "frank" else 2e-15
    for fam, rot, pars in CASE_KEYS:
        if (fam, rot) == (family, rotation):
            assert _ours(fam, rot, pars)["theta"] == pytest.approx(pars[0], rel=tol, abs=0.0)


@pytest.mark.parametrize("family, rotation", FAMILY_ROTATIONS,
                         ids=[f"{f}{r}" for f, r in FAMILY_ROTATIONS])
def test_scalar_paths_equal_array_paths(family, rotation):
    """pdf, cdf and inv_h: the scalar API is the array API bit for bit (measured)."""
    for fam, rot, pars in CASE_KEYS:
        if (fam, rot) != (family, rotation):
            continue
        cop, _ = _build(fam, rot, pars)
        ours = _ours(fam, rot, pars)
        np.testing.assert_array_equal([cop.pdf([a, b]) for a, b in POINTS], ours["pdf"])
        if ours["cdf"] is not None:
            np.testing.assert_array_equal([cop.cdf([a, b]) for a, b in POINTS], ours["cdf"])
        np.testing.assert_array_equal([cop.inv_h(b, a) for a, b in POINTS], ours["hinv1"])


# --------------------------------------------------------------------------
# Parity: pmcprg against each reference
# --------------------------------------------------------------------------

PARITY_IDS = [f"{n}-{f}{r}" for n in REFERENCES for f, r in FAMILY_ROTATIONS]
PARITY_PARAMS = [(n, f, r) for n in REFERENCES for f, r in FAMILY_ROTATIONS]


@pytest.mark.parametrize("reference, family, rotation", PARITY_PARAMS, ids=PARITY_IDS)
def test_interior_parity(reference, family, rotation):
    failures, compared = [], 0
    for key, case in _cases(reference).items():
        fam, rot, pars = key
        if (fam, rot) != (family, rotation):
            continue
        ours = _ours(fam, rot, pars)
        for q in VECTOR_QUANTITIES:
            ref = _ref_vector(case, q)
            if ref is None or ours[q] is None:
                continue
            err = _errors(q, ours[q], ref, reference)
            tol = _tolerance(reference, fam, rot, pars, q)
            compared += err.size
            if not err.max() <= tol:
                k = int(np.argmax(err))
                failures.append(f"{q} θ={pars} at {POINTS[k].tolist()}: pmcprg {ours[q][k]!r}, "
                                f"{reference} {ref[k]!r} (error {err[k]:.2e} > {tol:g})")
    assert compared > 0
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("reference, family, rotation", PARITY_PARAMS, ids=PARITY_IDS)
def test_tau_and_tail_dependence_parity(reference, family, rotation):
    """τ(θ), λ_L, λ_U — and pmcprg's θ(τ) map at the reference's τ."""
    failures = []
    for key, case in _cases(reference).items():
        fam, rot, pars = key
        if (fam, rot) != (family, rotation):
            continue
        ours = _ours(fam, rot, pars)
        for q in ("tau", "lambda_L", "lambda_U"):
            if case[q] is None:
                continue
            err = abs(ours[q] - case[q])
            tol = _tolerance(reference, fam, rot, pars, q)
            if not err <= tol:
                failures.append(f"{q} θ={pars}: pmcprg {ours[q]!r}, {reference} {case[q]!r}")
        if _limited(reference, fam, rot, pars, "tau") is None:
            # pmcprg's θ(τ) at the reference's τ(θ). Measured 1.6e-13 for
            # Frank at θ = 0.2 (the 3.8e-15 of τ over τ = 0.022), 7.8e-15
            # otherwise (Joe against copula).
            tol = 5e-13 if fam == "frank" else 2e-14
            cop, _ = _build(fam, rot, pars, tau=case["tau"])
            if not _theta(cop) == pytest.approx(pars[0], rel=tol, abs=0.0):
                failures.append(f"θ(τ_ref) θ={pars}: pmcprg {_theta(cop)!r}")
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("reference, family, rotation", PARITY_PARAMS, ids=PARITY_IDS)
def test_bivariate_law_conditions_on_either_margin(reference, family, rotation):
    """``BivariateLaw`` with uniform margins: its conditional laws are h1 and h2.

    P(V ≤ b | U = a) is the reference's h1(a, b), P(U ≤ a | V = b) its
    h2(a, b), and both conditional densities its c(a, b) — through the same
    tolerances as the copula itself (measured: the same maxima as
    :func:`test_interior_parity`, 4.2e-14 for h and 3.9e-14 relative for the
    density). Before the fix, the right-margin side swapped the copula's
    arguments and read h1(b, a) and c(b, a) instead: h2 off by up to 0.46
    (Joe 90°/270°), 0.42 (Clayton), 0.19 (Gumbel).
    """
    from scipy import stats

    from pmcprg.copulas import BivariateLaw
    from pmcprg.numerics import EPS, ONE_MINUS_EPS

    failures, compared = [], 0
    for key, case in _cases(reference).items():
        fam, rot, pars = key
        if (fam, rot) != (family, rotation):
            continue
        cop, tr = _build(fam, rot, pars)
        law = BivariateLaw(cop, (stats.uniform,), (stats.uniform,))
        ours = {
            "h1": np.array([law.conditional_cdf(b, a, which="left") for a, b in POINTS]),
            "h2": np.array([law.conditional_cdf(a, b, which="right") for a, b in POINTS]),
            # conditional_log_pdf floors the density at EPS (never reached
            # where the reference is above it)
            "pdf_left": np.array([law.conditional_pdf(b, a, which="left") for a, b in POINTS]),
            "pdf_right": np.array([law.conditional_pdf(a, b, which="right") for a, b in POINTS]),
        }
        for q, col in (("h1", "h1"), ("h2", "h2"), ("pdf_left", "pdf"), ("pdf_right", "pdf")):
            ref = _ref_vector(case, col)
            if ref is None:
                continue
            keep = ref > EPS if col == "pdf" else np.ones(ref.size, bool)
            err = _errors(col, ours[q][keep], ref[keep], reference)
            tol = _tolerance(reference, fam, rot, pars, col)
            compared += err.size
            if not err.max() <= tol:
                k = int(np.argmax(err))
                failures.append(f"{q} θ={pars} at {POINTS[keep][k].tolist()}: {err[k]:.2e} > {tol:g}")
        # Sampling the left margin given the right one inverts h2: the
        # transpose's inv_h on the law's own uniforms (hinv2 parity above).
        law.set_seed(20260921)
        b = float(POINTS[3, 1])
        drawn = law.sample_conditional(b, which="right", n=8)
        ws = np.random.default_rng(20260921).uniform(EPS, ONE_MINUS_EPS, 8)
        np.testing.assert_array_equal(drawn, tr.inv_h_array(ws, np.full(8, b)))
    assert compared > 0
    assert not failures, "\n".join(failures)


# --------------------------------------------------------------------------
# The references against one another
# --------------------------------------------------------------------------

# Largest mutual error of two references, outside REFERENCE_LIMITED (a cell
# is skipped when either reference is limited on it).
REFERENCE_TOL = {
    # 4.0e-14: pyvinecopulib against the two R packages (Gauss, Student).
    "pdf": 1e-13,
    # 1.6e-15: VineCopula against copula (Clayton θ = 9).
    "cdf": 5e-15,
    # 4.2e-14: copula's Frank θ = −8 against VineCopula; 1.8e-14
    # pyvinecopulib against copula (Joe).
    "h1": 1e-13,
    "h2": 1e-13,
    # 2.4e-15: VineCopula against copula (Clayton).
    "hinv1": 1e-14,
    "hinv2": 1e-14,
    # 4.0e-15: pyvinecopulib against copula (Joe).
    "tau": 1e-14,
    # 2.5e-16 (Student).
    "lambda_L": 1e-15,
    "lambda_U": 1e-15,
}
PAIRS = [("pyvinecopulib", "vinecopula"), ("pyvinecopulib", "rcopula"), ("vinecopula", "rcopula")]


@pytest.mark.parametrize("a, b", PAIRS, ids=[f"{a}-{b}" for a, b in PAIRS])
def test_references_agree_with_each_other(a, b):
    failures, compared = [], 0
    for key in CASE_KEYS:
        fam, rot, pars = key
        ca, cb = _cases(a)[key], _cases(b)[key]
        for q in VECTOR_QUANTITIES + ("tau", "lambda_L", "lambda_U"):
            if ca[q] is None or cb[q] is None:
                continue
            if _limited(a, fam, rot, pars, q) or _limited(b, fam, rot, pars, q):
                continue
            x, y = np.atleast_1d(np.array(ca[q], dtype=float)), np.atleast_1d(np.array(cb[q], dtype=float))
            if q in ("h1", "h2") and "vinecopula" in (a, b):
                x, y = np.clip(x, *VINECOPULA_H_RANGE), np.clip(y, *VINECOPULA_H_RANGE)
            err = np.abs(x - y) / (np.abs(y) if q == "pdf" else 1.0)
            compared += err.size
            if not err.max() <= REFERENCE_TOL[q]:
                failures.append(f"{fam}/{rot} θ={pars} {q}: {err.max():.2e}")
    assert compared > 0
    assert not failures, "\n".join(failures)


# --------------------------------------------------------------------------
# Discrepancies, arbitrated by mpmath
# --------------------------------------------------------------------------

def _mp_copula_cdf(family: str, rotation: int, pars: tuple):
    """The textbook CDF (Nelsen 2006, Table 4.1; Joe 1997), rotated as measured."""
    th = mp.mpf(pars[0])
    if family == "clayton":
        def C(u, v):
            return (u ** -th + v ** -th - 1) ** (-1 / th)
    elif family == "gumbel":
        def C(u, v):
            return mp.exp(-(((-mp.log(u)) ** th + (-mp.log(v)) ** th) ** (1 / th)))
    elif family == "joe":
        def C(u, v):
            a, b = (1 - u) ** th, (1 - v) ** th
            return 1 - (a + b - a * b) ** (1 / th)
    elif family == "frank":
        def C(u, v):
            return -mp.log(1 + mp.expm1(-th * u) * mp.expm1(-th * v) / mp.expm1(-th)) / th
    else:
        raise KeyError(family)
    return {0: C,
            180: lambda u, v: u + v - 1 + C(1 - u, 1 - v),
            90: lambda u, v: v - C(1 - u, v),
            270: lambda u, v: u - C(u, 1 - v)}[rotation]


def _mp_root(f, start: float):
    x0 = mp.mpf(start)
    root = mp.findroot(f, (x0, x0 * (1 + mp.mpf("1e-9"))), solver="secant")
    assert abs(f(root)) < mp.mpf(10) ** -30
    return root


def _mp_value(family: str, rotation: int, pars: tuple, q: str, a: float, b: float, start: float):
    """q at the point (a, b), from the CDF alone: derivatives by mp.diff, inverses by root finding."""
    C = _mp_copula_cdf(family, rotation, pars)
    u, v = mp.mpf(a), mp.mpf(b)

    def h1(x, y):
        return mp.diff(lambda s: C(s, y), x)

    def h2(x, y):
        return mp.diff(lambda s: C(x, s), y)

    if q == "cdf":
        return C(u, v)
    if q == "pdf":
        return mp.diff(C, (u, v), (1, 1))
    if q == "h1":
        return h1(u, v)
    if q == "h2":
        return h2(u, v)
    if q == "hinv1":
        return _mp_root(lambda y: h1(u, y) - v, start)
    if q == "hinv2":
        return _mp_root(lambda x: h2(x, v) - u, start)
    raise KeyError(q)


def _mp_frank_tau(theta: float):
    """τ = 1 − 4/θ + (4/θ²) ∫₀^θ t/(eᵗ − 1) dt (Genest 1987)."""
    th = mp.mpf(theta)
    return 1 - 4 / th + 4 / th ** 2 * mp.quad(lambda t: t / mp.expm1(t), [0, th])


def _worst(entry: Limited, q: str):
    """(error, key, point index) of the largest |pmcprg − reference| in the entry's scope."""
    worst = (-1.0, None, None)
    for key, case in _cases(entry.reference).items():
        fam, rot, pars = key
        if not entry.covers(entry.reference, fam, rot, pars, q):
            continue
        ours = _ours(fam, rot, pars)
        if q == "tau":
            err, k = abs(ours["tau"] - case["tau"]), None
        else:
            e = _errors(q, ours[q], _ref_vector(case, q), entry.reference)
            k = int(np.argmax(e))
            err = float(e[k])
        if err > worst[0]:
            worst = (err, key, k)
    return worst


LIMITED_PARAMS = [(e, q) for e in REFERENCE_LIMITED for q in e.quantities]


@pytest.mark.parametrize(
    "entry, q", LIMITED_PARAMS,
    ids=[f"{e.reference}-{e.family}{''.join(map(str, e.rotations))}-"
         f"{'all' if e.pars is None else e.pars[0][0]}-{q}" for e, q in LIMITED_PARAMS])
def test_reference_limited_entries_are_the_reference(entry, q):
    """At the worst point of each entry, mpmath sides with pmcprg, not with the reference."""
    pytest.importorskip("mpmath")
    err, key, k = _worst(entry, q)
    assert key is not None
    # Still needed: the reference is off by more than the default tolerance
    # (else the entry must go), and by no more than the entry allows.
    assert TOL[q] < err <= entry.tol, (err, entry)
    fam, rot, pars = key
    ours = _ours(fam, rot, pars)
    with mp.workdps(40):
        if q == "tau":
            exact = _mp_frank_tau(pars[0])
            ours_value, ref_value = ours["tau"], _cases(entry.reference)[key]["tau"]
        else:
            a, b = POINTS[k]
            exact = _mp_value(fam, rot, pars, q, a, b, start=ours[q][k])
            ours_value, ref_value = ours[q][k], _cases(entry.reference)[key][q][k]
        scale = abs(exact) if q == "pdf" else 1
        ours_err = float(abs(ours_value - exact) / scale)
        ref_err = float(abs(ref_value - exact) / scale)
    # pmcprg within the default tolerance of the exact value; the reference
    # carries the discrepancy (measured: pmcprg ≤ 4.7e-16 absolute and
    # ≤ 1.2e-14 relative at these points, the references 10⁻¹² to 10⁻³).
    assert ours_err <= TOL[q], (ours_err, entry)
    assert ref_err >= 0.9 * err, (ref_err, err, entry)


def test_gaussian_cdf_gap_is_pmcprg_quadrature():
    """The one PMCPRG_LIMITED cell: the references agree with mpmath, pmcprg is 1.2e-14 off.

    Φ_ρ(x, y) = uv + (1/2π) ∫₀^ρ exp(−(x² − 2rxy + y²)/(2(1 − r²)))/√(1 − r²) dr
    (Drezner & Wesolowsky 1990) in 40-digit mpmath. pmcprg integrates the same
    kind of one-dimensional integral with a composite 8-point Gauss–Legendre
    rule whose documented relative error is below 2·10⁻¹⁰
    (``pmcprg/copulas/elliptical/gaussian.py``); Genz's BVND, which the
    references use, reaches 10⁻¹⁶. Measured at the worst point: pmcprg
    1.15e-14 (7.1e-11 relative), each reference ≤ 1.1e-16.
    """
    pytest.importorskip("mpmath")
    worst = (-1.0, None, None)
    for key in CASE_KEYS:
        if key[0] != "gaussian":
            continue
        e = np.abs(_ours(*key)["cdf"] - np.array(_cases("pyvinecopulib")[key]["cdf"]))
        k = int(np.argmax(e))
        if e[k] > worst[0]:
            worst = (float(e[k]), key, k)
    err, key, k = worst
    assert TOL["cdf"] < err <= PMCPRG_LIMITED[("gaussian", "cdf")]
    rho = mp.mpf(key[2][0])
    a, b = POINTS[k]
    with mp.workdps(40):
        u, v = mp.mpf(a), mp.mpf(b)
        x, y = mp.sqrt(2) * mp.erfinv(2 * u - 1), mp.sqrt(2) * mp.erfinv(2 * v - 1)
        exact = u * v + mp.quad(lambda r: mp.exp(-(x * x - 2 * r * x * y + y * y) / (2 * (1 - r * r)))
                                / mp.sqrt(1 - r * r), [0, rho]) / (2 * mp.pi)
        ours_err = float(abs(_ours(*key)["cdf"][k] - exact))
        rel = ours_err / float(exact)
        ref_errs = [float(abs(_cases(n)[key]["cdf"][k] - exact)) for n in REFERENCES]
    assert ours_err > 20 * max(ref_errs)            # the gap is pmcprg's
    assert rel < 2e-10                              # within gaussian.py's documented accuracy
    assert max(ref_errs) < 5e-16
