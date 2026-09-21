"""Interior parity of Joe's BB1, BB6, BB7 and BB8 — audit FR-11, wave 2.

BB1 in its four rotations — ``BB1``, ``BB190``, ``SURVIVAL_BB1`` (180°),
``BB1270``, and the survival's own rotations ``SURVIVAL_BB190`` and
``SURVIVAL_BB1270``, which are BB1's 270° and 90° rotations — and BB6, BB7,
BB8 unrotated (pmcprg registers no rotation of these three): 23 parameter
sets on the 25 interior points of the pilot (``scripts/parity/cases_bb.csv``,
``points.csv``), against

* pyvinecopulib 1.0.0 (vinecopulib, C++) — every quantity;
* VineCopula 2.6.1 (families 7–10, 17, 27, 37) — every quantity;
* mpmath, everywhere.

R ``copula``, copBasic and fCopulae have none of the four families (copBasic
has BB4 only), and the two vine libraries share their formulas — they agree
with each other to 3.2·10⁻¹⁴ outside the register below, and are mostly
wrong together where they are wrong. The **mpmath oracle** is therefore
what makes each family rest on more than one implementation: it evaluates
every quantity of every case on every point from the textbook CDF and
generator of Joe (1997, ch. 5) alone — pdf and h by ``mp.diff``, h⁻¹ by root
finding, τ by the Genest–MacKay integral 1 + 4∫φ/φ′ (≤ 2.2·10⁻²⁰ from the
literature's closed forms: BB1's 1 − 2/(δ(θ+2)), BB6's through Joe's digamma
form, BB7's Beta form, BB8's ₃F₂ — measured, not used), λ by the diagonal
limits.

The tables are generated offline by ``scripts/parity/gen_pyvinecopulib_bb.py``
and ``scripts/parity/gen_r_bb.R`` (``scripts/parity/README.md``); nothing here
needs R or pyvinecopulib.

Convention mapping (measured by the generators, re-read by
:func:`test_recorded_conventions`)
-----------------------------------------------------------------------------
* **Key.** ``[θ, δ]`` of the unrotated family, as in Joe (1997, ch. 5):
  BB1 (θ > 0, δ ≥ 1) C = (1 + ((u^−θ − 1)^δ + (v^−θ − 1)^δ)^{1/δ})^{−1/θ};
  BB6 (θ ≥ 1, δ ≥ 1) C = 1 − (1 − exp(−(p^δ + q^δ)^{1/δ}))^{1/θ},
  p = −log(1 − (1 − u)^θ); BB7 (θ ≥ 1, δ > 0)
  C = 1 − (1 − (a^−δ + b^−δ − 1)^{−1/δ})^{1/θ}, a = 1 − (1 − u)^θ; BB8
  (θ ≥ 1, 0 < δ ≤ 1) C = (1 − (1 − A)^{1/θ})/δ,
  A = (1 − (1 − δu)^θ)(1 − (1 − δv)^θ)/(1 − (1 − δ)^θ). Both packages take
  ``[θ, δ]`` in that order — their CDF is the textbook one (median ≤ 2.8·10⁻¹⁷
  from a double-precision evaluation of it; at most 3.7·10⁻¹⁰, where that
  evaluation itself cancels), the swapped order, where their bounds accept
  it at all, does not match — and so do pmcprg's modules
  (``bb1.py``: θ, ``delta``; ``bb6.py``: θ, ``delta6``; ``bb7.py``:
  ``theta7``, δ; ``bb8.py``: θ, ``delta8``), whose τ maps the mpmath oracle
  checks independently.
* **Rotations.** 90° reflects u, 270° v, 180° both (pilot); pyvinecopulib
  takes ``rotation=`` with the positive parameters, VineCopula families
  27/37 **both** parameters negated (−θ, −δ), 17 positive (measured). The
  survival's 90°/270° rotations of pmcprg are BB1's 270°/90° ones
  (``rotated.py``), compared with the same case.
* **h-functions** as in the pilot: h1 = ``conditional_cdf(v, u)``, h2 the
  transposed copula's ``conditional_cdf(u, v)`` (90°ᵀ = 270°, the others
  exchangeable); VineCopula's h compared clamped to [10⁻¹², 1 − 10⁻¹²].
* **Tail dependence**: the diagonal pair; (0, 0) for 90°/270° in all three.
* **pmcprg at the key's parameters.** pmcprg is built from τ by its own
  τ(θ, δ) — BB1 and BB6 closed forms, BB7's series, BB8's quadrature — and
  its constructor's inverse is tested on its own
  (:func:`test_pmcprg_parameter_round_trip`: the recovered θ — δ for BB7 —
  to 2.0·10⁻¹⁵, BB8's Brent search to 1.2·10⁻¹⁴); the values are then
  evaluated at the key's exact (θ, δ), assigned to the built copula and
  every wrapper under it — they test the formulas, not the inversion.

Tolerances and discrepancies
----------------------------
:data:`TOL` (pmcprg against a package) and :data:`ORACLE_TOL` (pmcprg against
mpmath) are small multiples of the largest error measured on the grid,
stated next to each. Outside :data:`PMCPRG_LIMITED`, pmcprg agrees with
mpmath to 1.3·10⁻¹⁴ (pdf, relative), 2.0·10⁻¹⁶ (cdf), 9.5·10⁻¹⁵ (h),
5.7·10⁻¹⁶ (h⁻¹, BB8), 1.7·10⁻¹⁶ (τ; 6.1·10⁻¹⁶ under numpy 1.24 / scipy
1.10, the rounding of BB8's quadrature sum), 1.1·10⁻¹⁶ (λ) — the same
figures under the minimum versions but for that one.

* :data:`PMCPRG_LIMITED` — pmcprg's h⁻¹, 4.4·10⁻¹⁵ to 1.7·10⁻¹⁴. Where h
  is flat (w = 0.99, small density) the root is only defined to ε·w/c, the
  rounding of h itself: BB1's generic Brent search (``CopulaVirt.inv_h``,
  xtol 10⁻¹⁵) and the logit bisections of BB6 and BB7 land within 11 times
  that floor (BB8's within 1.5 times, inside :data:`ORACLE_TOL`). The BB1
  **rotations and survival** use their kernel inverse
  (``bb1._bb1_inv_h_k``, Brent on log v): its ``xtol = 1e-13`` left
  1.5·10⁻¹⁴, up to 293 times the floor; tightened to 1e-15 at the author's
  decision (same cost), it is 7.1·10⁻¹⁵.
* :data:`REFERENCE_LIMITED` — a package is off and pmcprg agrees with mpmath,
  each entry proven at its worst point
  (:func:`test_reference_limited_entries_are_the_reference`): both vine
  libraries' numerical h-inverses (pyvinecopulib 2.9·10⁻¹¹, VineCopula
  7.2·10⁻¹²; up to 6.6·10⁻¹⁰ and 6.0·10⁻⁹ where their h is itself off); the
  textbook formulas they evaluate in linear scale, which cancel near (1, 1)
  — BB6 (5, 1.5) density 3.5·10⁻⁷ relative, BB7 (4, 3) 4.9·10⁻⁹, BB8's Joe
  edge δ = 1 CDF 1.5·10⁻¹¹ and pyvinecopulib's h there 2.0·10⁻⁹,
  VineCopula's BB6 (5, 1.5) h 7.5·10⁻⁸; VineCopula's τ of BB6, BB7, BB8
  (numerical integration, ≤ 1.1·10⁻⁷).

The packages' mutual agreement outside these entries is asserted by
:func:`test_references_agree_with_each_other`.
"""
from __future__ import annotations

import functools
import json
import platform
import sys
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

try:                              # a dev dependency: the minimum-versions CI job
    import mpmath as mp           # installs the runtime ones only, and the table
except ImportError:               # comparisons must still run there
    mp = None
import numpy as np
import pytest

_BIT_EXACT = sys.platform == "darwin" and platform.machine() == "arm64"

from pmcprg.copulas import CopulaEnum
from pmcprg.copulas.archimedean import bb8
from pmcprg.copulas.archimedean.bb7 import _bb7_tau_from_theta
from pmcprg.copulas.archimedean.joe import _joe_tau_from_theta

DATA = Path(__file__).resolve().parent / "data" / "parity"
REFERENCES = ("pyvinecopulib", "vinecopula")
VECTOR_QUANTITIES = ("pdf", "cdf", "h1", "h2", "hinv1", "hinv2")
SCALAR_QUANTITIES = ("tau", "lambda_L", "lambda_U")
HINV = ("hinv1", "hinv2")
H = ("h1", "h2")


@functools.cache
def _table(name: str) -> dict:
    return json.loads((DATA / f"bb_{name}.json").read_text())


@functools.cache
def _cases(name: str) -> dict:
    return {(c["family"], c["rotation"], tuple(c["pars"])): c for c in _table(name)["cases"]}


POINTS = np.array(_table("vinecopula")["points"], dtype=float)
CASE_KEYS = tuple((f, r, tuple(p)) for f, r, p in _table("vinecopula")["grid"])
FAMILY_ROTATIONS = tuple(dict.fromkeys((f, r) for f, r, _ in CASE_KEYS))
FR_IDS = [f"{f}{r}" for f, r in FAMILY_ROTATIONS]

# VineCopula returns its h-functions clamped to this interval (pilot).
VINECOPULA_H_RANGE = (1e-12, 1.0 - 1e-12)

# The pmcprg members a case is compared with (module docstring).
MEMBERS = {("bb1", 0): ("BB1",), ("bb1", 90): ("BB190", "SURVIVAL_BB1270"),
           ("bb1", 180): ("SURVIVAL_BB1",), ("bb1", 270): ("BB1270", "SURVIVAL_BB190"),
           ("bb6", 0): ("BB6",), ("bb7", 0): ("BB7",), ("bb8", 0): ("BB8",)}
TRANSPOSE = {"BB190": "BB1270", "BB1270": "BB190",
             "SURVIVAL_BB190": "SURVIVAL_BB1270", "SURVIVAL_BB1270": "SURVIVAL_BB190"}
# The name under which each family takes its second parameter, and the
# attribute of the (innermost) copula holding the one it *recovers* from τ.
EXTRA = {"bb1": "delta", "bb6": "delta6", "bb7": "theta7", "bb8": "delta8"}
RECOVERED = {"bb1": "theta", "bb6": "theta", "bb7": "delta7", "bb8": "theta"}


def _member_params():
    return [(m, f, r, p) for f, r, p in CASE_KEYS for m in MEMBERS[(f, r)]]


# --------------------------------------------------------------------------
# pmcprg side
# --------------------------------------------------------------------------

def _tau_of(family: str, pars: tuple) -> float:
    """pmcprg's own τ of the unrotated family at the key's (θ, δ)."""
    th, de = pars
    if family == "bb1":
        return 1.0 - 2.0 / (de * (th + 2.0))                # bb1.py, inverted by its constructor
    if family == "bb6":
        return 1.0 - (1.0 - _joe_tau_from_theta(th)) / de   # CopulaBB6.tau_of
    if family == "bb7":
        return _bb7_tau_from_theta(th, de)
    if family == "bb8":
        return bb8._bb8_tau_quad(th, de)                    # bb8._tau_of, without its memo
    raise KeyError(family)


def _build(member: str, family: str, rotation: int, pars: tuple, tau: float | None = None):
    """pmcprg's copula at τ (by default pmcprg's τ of the key's parameters).

    BB8's constructor answers a τ its forward map produced earlier in the
    process from a memo (``bb8.py``); it is built on an empty memo here, so
    that its θ is its Brent search's whatever ran before.
    """
    if tau is None:
        tau = _tau_of(family, pars) * (-1.0 if rotation in (90, 270) else 1.0)
    kwargs = {"tau_k": tau, EXTRA[family]: pars[0] if family == "bb7" else pars[1]}
    klass = getattr(CopulaEnum, member).klass
    if family == "bb8":
        with mock.patch.object(bb8, "_THETA_MEMO", OrderedDict()):
            return klass(**kwargs)
    return klass(**kwargs)


def _innermost(cop):
    while getattr(cop, "_base", None) is not None:
        cop = cop._base
    return cop


def _recovered(cop, family: str) -> float:
    return getattr(_innermost(cop), RECOVERED[family])


def _pin(cop, family: str, pars: tuple) -> None:
    """Evaluate at exactly the key's (θ, δ) (module docstring): the recovered
    parameter is assigned on every level of the wrapper chain — the kernels
    read the innermost's (``SURVIVAL_BB190`` wraps a ``SurvivalBB1`` that
    wraps a ``CopulaBB1``)."""
    value = pars[1] if family == "bb7" else pars[0]
    while cop is not None:
        setattr(cop, RECOVERED[family], value)
        cop = getattr(cop, "_base", None)


@functools.cache
def _pair(member: str, family: str, rotation: int, pars: tuple):
    """(copula, its ``transposed()``, the parameter its constructor recovered), at the key's (θ, δ)."""
    cop = _build(member, family, rotation, pars)
    built = _recovered(cop, family)
    tr = cop.transposed()
    assert type(tr) is getattr(CopulaEnum, TRANSPOSE.get(member, member)).klass
    _pin(cop, family, pars)
    if tr is not cop:
        _pin(tr, family, pars)
    return cop, tr, built


@functools.cache
def _ours(member: str, family: str, rotation: int, pars: tuple) -> dict:
    cop, tr, built = _pair(member, family, rotation, pars)
    u, v = POINTS[:, 0].copy(), POINTS[:, 1].copy()
    lam_l, lam_u = cop.tail_dependence()
    return {
        "built": built,
        "pdf": cop.pdf_array(POINTS),
        "cdf": cop.cdf_array(POINTS),
        "h1": np.array([cop.conditional_cdf(b, a) for a, b in POINTS]),
        "h2": np.array([tr.conditional_cdf(a, b) for a, b in POINTS]),
        "hinv1": cop.inv_h_array(v, u),
        "hinv2": tr.inv_h_array(u, v),
        "tau": float(cop.params["tau_k"]),
        "lambda_L": lam_l,
        "lambda_U": lam_u,
    }


def _ref_vector(case: dict, q: str) -> np.ndarray | None:
    values = case.get(q)
    if values is None:
        return None
    assert None not in values, f"partial null in {case['native']} {q}"
    return np.array(values, dtype=float)


def _errors(q: str, x, ref, reference: str) -> np.ndarray:
    """|x − ref|, relative for the density; VineCopula's h compared clamped."""
    x = np.asarray(x, dtype=float)
    if reference == "vinecopula" and q in H:
        x = np.clip(x, *VINECOPULA_H_RANGE)
    d = np.abs(x - ref)
    return d / np.abs(ref) if q == "pdf" else d


# --------------------------------------------------------------------------
# Tolerances (module docstring). Absolute; the density's relative.
# --------------------------------------------------------------------------

# pmcprg against a package, outside the registers below.
TOL = {
    # 2.8e-14: both packages, BB7 (1.3, 0.5) at (0.99, 0.99), themselves
    # 2.9e-14 from mpmath there (pmcprg 1.3e-14 at most, BB1 (2.5, 3)).
    "pdf": 5e-14,
    # 2.8e-15: VineCopula's BB8 (3, 0.9) at (0.99, 0.99), its own error 2.7e-15.
    "cdf": 5e-15,
    # 2.9e-14: pyvinecopulib's BB7 (1.5, 5) at (0.01, 0.5), its own error 3.2e-14.
    "h1": 5e-14,
    "h2": 5e-14,
    # No accurate package inverse (REFERENCE_LIMITED): unused, pmcprg's own
    # errors are bounded by ORACLE_TOL and PMCPRG_LIMITED.
    "hinv1": 5e-14,
    "hinv2": 5e-14,
    # 4.2e-16: pyvinecopulib's BB8 (1.5, 0.6) (pyvinecopulib 2.5e-16 and
    # pmcprg 1.7e-16 from mpmath); 8.6e-16 under numpy 1.24 / scipy 1.10,
    # where pmcprg's quadrature ends 6.1e-16 from mpmath (ORACLE_TOL).
    "tau": 2e-15,
    # 0 measured: both packages and pmcprg evaluate the same closed forms.
    "lambda_L": 5e-16,
    "lambda_U": 5e-16,
}

# pmcprg against mpmath, outside PMCPRG_LIMITED.
ORACLE_TOL = {
    # 1.3e-14: BB1 (2.5, 3) 90° at (0.25, 0.97).
    "pdf": 5e-14,
    # 2.0e-16: BB6 (5, 1.5).
    "cdf": 5e-16,
    # 9.5e-15: BB1 (2.5, 3) 90° at (0.25, 0.97).
    "h1": 2e-14,
    "h2": 2e-14,
    # 5.7e-16: BB8 (3, 0.9) at (0.5, 0.99), the bisection's rounding floor.
    "hinv1": 2e-15,
    "hinv2": 2e-15,
    # 1.7e-16: BB8 (1.5, 0.6), the graded Gauss–Legendre quadrature of
    # bb8.py (τ = 0.059); 6.1e-16 under numpy 1.24 / scipy 1.10, the
    # rounding of that sum.
    "tau": 1e-15,
    # 1.1e-16: BB1 (1.2, 1.5), BB7 (1.5, 5).
    "lambda_L": 5e-16,
    "lambda_U": 5e-16,
}

# pmcprg is the one off (module docstring), measured against mpmath:
# (family, rotation, quantity) → tolerance.
PMCPRG_LIMITED = {
    # The generic brentq of CopulaVirt.inv_h (xtol 1e-15): 6.4e-15 at
    # (u, w) = (0.01, 0.99), (θ, δ) = (1.2, 1.5), 3.9 times ε·w/c with c = 0.13.
    **{("bb1", 0, q): 1e-14 for q in HINV},
    # bb1._bb1_inv_h_k, Brent on log v, xtol 1e-15 since FR-11 wave 2 (it was
    # 1e-13 and left 1.5e-14): 7.1e-15 ((1.2, 1.5) at (0.01, 0.01)), the
    # rounding floor ε·w/c of h where it is flat, as for the generic search.
    **{("bb1", r, q): 2e-14 for r in (90, 180, 270) for q in HINV},
    # The logit bisection, at the rounding floor where h is flat: 4.4e-15
    # (BB6 (5, 1.5) at (0.5, 0.99), 5.4 times the floor), 1.7e-14 (BB7
    # (2, 1.5) at (0.01, 0.99), c = 0.068, 5.3 times the floor).
    **{("bb6", 0, q): 1e-14 for q in HINV},
    **{("bb7", 0, q): 5e-14 for q in HINV},
}


@dataclass(frozen=True)
class Limited:
    """A cell where the *package* is off — its tolerance is the package's error.

    ``pars`` None: every parameter set of the family (every rotation). Proven
    by :func:`test_reference_limited_entries_are_the_reference` with mpmath
    at its worst point, and must still be needed (worst error above the
    default).
    """

    reference: str
    family: str
    pars: tuple | None
    quantities: tuple
    tol: float
    why: str

    def covers(self, reference: str, family: str, pars: tuple, q: str) -> bool:
        return (reference == self.reference and family == self.family and q in self.quantities
                and (self.pars is None or tuple(pars) in self.pars))


_BB6_WEAK = ((1.3, 1.2), (2.0, 1.8))
_BB7_BUT_43 = ((1.3, 0.5), (2.0, 1.5), (1.5, 5.0))

REFERENCE_LIMITED = (
    # --- the numerical inverses of both vine libraries
    Limited("pyvinecopulib", "bb1", None, HINV, 6e-11, "numerical inversion: 2.9e-11"),
    Limited("pyvinecopulib", "bb6", None, HINV, 6e-11, "numerical inversion: 2.9e-11"),
    Limited("pyvinecopulib", "bb7", _BB7_BUT_43, HINV, 6e-11, "numerical inversion: 2.9e-11"),
    Limited("pyvinecopulib", "bb8", None, HINV, 6e-11, "numerical inversion: 3.0e-11"),
    Limited("vinecopula", "bb1", None, HINV, 1.5e-11, "numerical inversion: 6.7e-12"),
    Limited("vinecopula", "bb6", _BB6_WEAK, HINV, 1.5e-11, "numerical inversion: 6.1e-12"),
    Limited("vinecopula", "bb7", _BB7_BUT_43, HINV, 1.5e-11, "numerical inversion: 7.2e-12"),
    Limited("vinecopula", "bb8", None, HINV, 1.5e-11, "numerical inversion: 6.8e-12"),
    # --- the textbook formulas in linear scale, cancelling near (1, 1)
    Limited("pyvinecopulib", "bb6", ((2.0, 1.8),), ("pdf",), 2e-13,
            "density at (0.01, 0.99): 9.0e-14 relative"),
    Limited("vinecopula", "bb6", ((2.0, 1.8),), ("pdf",), 2e-13,
            "density at (0.99, 0.5): 9.0e-14 relative"),
    Limited("pyvinecopulib", "bb6", ((5.0, 1.5),), ("pdf",), 1e-6,
            "density at (0.99, 0.99): 3.5e-7 relative"),
    Limited("vinecopula", "bb6", ((5.0, 1.5),), ("pdf",), 1e-6,
            "density at (0.99, 0.99): 3.5e-7 relative"),
    Limited("vinecopula", "bb6", ((5.0, 1.5),), ("cdf",), 1e-9,
            "CDF at (0.99, 0.99): 3.7e-10 (pyvinecopulib's is right there)"),
    Limited("vinecopula", "bb6", ((5.0, 1.5),), H, 2e-7, "h at (0.99, 0.99): 7.5e-8"),
    Limited("vinecopula", "bb6", ((5.0, 1.5),), HINV, 1.5e-8,
            "its inverse of that h, at (0.99, 0.99): 6.0e-9"),
    Limited("pyvinecopulib", "bb7", ((2.0, 1.5),), ("pdf",), 1e-12,
            "density at (0.99, 0.99): 5.2e-13 relative"),
    Limited("vinecopula", "bb7", ((2.0, 1.5),), ("pdf",), 1e-12,
            "density at (0.99, 0.99): 5.2e-13 relative"),
    Limited("pyvinecopulib", "bb7", ((2.0, 1.5),), H, 3e-13, "h at (0.99, 0.99): 1.2e-13"),
    Limited("vinecopula", "bb7", ((2.0, 1.5),), H, 3e-13, "h at (0.99, 0.99): 1.2e-13"),
    Limited("pyvinecopulib", "bb7", ((4.0, 3.0),), ("pdf",), 1e-8,
            "density at (0.99, 0.99): 4.9e-9 relative"),
    Limited("vinecopula", "bb7", ((4.0, 3.0),), ("pdf",), 1e-8,
            "density at (0.99, 0.99): 4.9e-9 relative"),
    Limited("pyvinecopulib", "bb7", ((4.0, 3.0),), ("cdf",), 2e-11, "CDF at (0.99, 0.99): 8.4e-12"),
    Limited("vinecopula", "bb7", ((4.0, 3.0),), ("cdf",), 2e-11, "CDF at (0.99, 0.99): 8.4e-12"),
    Limited("pyvinecopulib", "bb7", ((4.0, 3.0),), H, 3e-9, "h at (0.99, 0.99): 1.3e-9"),
    Limited("vinecopula", "bb7", ((4.0, 3.0),), H, 3e-9, "h at (0.99, 0.99): 1.3e-9"),
    Limited("pyvinecopulib", "bb7", ((4.0, 3.0),), HINV, 1.5e-9,
            "its inverse of that h, at (0.99, 0.99): 6.6e-10"),
    Limited("vinecopula", "bb7", ((4.0, 3.0),), HINV, 3e-9,
            "its inverse of that h, at (0.99, 0.99): 1.4e-9"),
    Limited("pyvinecopulib", "bb8", ((4.0, 1.0),), ("cdf",), 3e-11,
            "the Joe edge δ = 1, CDF at (0.99, 0.99): 1.3e-11"),
    Limited("vinecopula", "bb8", ((4.0, 1.0),), ("cdf",), 3e-11,
            "the Joe edge δ = 1, CDF at (0.99, 0.99): 1.5e-11"),
    Limited("pyvinecopulib", "bb8", ((4.0, 1.0),), H, 5e-9,
            "the Joe edge δ = 1, h at (0.99, 0.99): 2.0e-9 (VineCopula's is right there)"),
    # --- VineCopula's τ by numerical integration (pyvinecopulib's within 2.5e-16)
    Limited("vinecopula", "bb6", None, ("tau",), 3e-7, "integrate(): 1.1e-7 at (5, 1.5)"),
    Limited("vinecopula", "bb7", None, ("tau",), 6e-8, "integrate(): 2.6e-8 at (1.3, 0.5)"),
    Limited("vinecopula", "bb8", None, ("tau",), 2e-7, "integrate(): 9.1e-8"),
)


def _limited(reference: str, family: str, pars: tuple, q: str) -> Limited | None:
    hits = [e for e in REFERENCE_LIMITED if e.covers(reference, family, pars, q)]
    assert len(hits) <= 1, hits
    return hits[0] if hits else None


def _tolerance(reference: str, family: str, rotation: int, pars: tuple, q: str) -> float:
    entry = _limited(reference, family, pars, q)
    if entry is not None:
        return entry.tol
    return PMCPRG_LIMITED.get((family, rotation, q), TOL[q])


def _oracle_tolerance(family: str, rotation: int, q: str) -> float:
    return PMCPRG_LIMITED.get((family, rotation, q), ORACLE_TOL[q])


# --------------------------------------------------------------------------
# The tables
# --------------------------------------------------------------------------

def test_tables_share_grid_and_carry_provenance():
    root = Path(__file__).resolve().parents[2] / "scripts" / "parity"
    points = np.loadtxt(root / "points.csv", delimiter=",", skiprows=1)
    np.testing.assert_array_equal(POINTS, points)
    assert POINTS.min() >= 0.01 and POINTS.max() <= 0.99          # interior only
    grid = np.loadtxt(root / "cases_bb.csv", delimiter=",", skiprows=1, usecols=(1, 2, 3))
    assert [(r, (a, b)) for _, r, (a, b) in CASE_KEYS] == [(int(r), (a, b)) for r, a, b in grid]
    for name in REFERENCES:
        t = _table(name)
        np.testing.assert_array_equal(np.array(t["points"], dtype=float), POINTS)
        assert tuple((f, r, tuple(p)) for f, r, p in t["grid"]) == CASE_KEYS
        assert set(_cases(name)) == set(CASE_KEYS)        # both packages cover the whole grid
        for key in ("reference", "environment", "generated", "repository_version",
                    "command", "definitions", "convention_checks"):
            assert t[key], (name, key)
        assert t["reference"]["version"]
    assert set(FAMILY_ROTATIONS) == set(MEMBERS)


EXPECTED_REFLECTION = {180: "c0(1-u, 1-v)", 90: "c0(1-u, v)", 270: "c0(u, 1-v)"}
EXPECTED_ORDER = {"pyvinecopulib": "[theta, delta]", "vinecopula": "par = theta, par2 = delta"}


@pytest.mark.parametrize("name", REFERENCES)
def test_recorded_conventions(name):
    """The parametrisation and rotations of the module docstring are the ones the generators measured."""
    checks = _table(name)["convention_checks"]
    assert checks["parameter_order"] == {f: EXPECTED_ORDER[name] for f in ("bb1", "bb6", "bb7", "bb8")}
    assert set(checks["rotations"]) == {"bb1/90", "bb1/180", "bb1/270"}
    for key, rc in checks["rotations"].items():
        rotation = int(key.split("/")[1])
        assert rc["density_equals"] == EXPECTED_REFLECTION[rotation], (name, key)
        assert rc["tau_rotated_over_tau_base"] == pytest.approx(-1.0 if rotation in (90, 270) else 1.0)
        if name == "vinecopula":
            # Families 27/37 take (−θ, −δ), 17 (θ, δ).
            assert rc["parameter_sign"] == ("positive" if rotation == 180 else "negative")
    for key, fam_checks in checks["per_family"].items():
        for check, value in fam_checks.items():
            if not check.endswith("_median"):
                continue
            if check.startswith("param_"):
                tol = checks["param_tolerance"]
            elif "hinv" in check:
                tol = checks["roundtrip_tolerance"]
            else:
                tol = checks["fd_tolerance"]
            assert value < tol, (name, key, check, value)
        # The textbook CDF (a double-precision evaluation of it) to 1.6e-12
        # (VineCopula's BB8 at δ = 1) and 3.7e-10 (pyvinecopulib's BB6
        # (5, 1.5) at (0.99, 0.99), where the double-precision textbook
        # formula itself cancels: pyvinecopulib is 8.7e-17 from mpmath there).
        assert fam_checks["param_cdf_abs_error_max"] < 1e-9, (name, key)


# --------------------------------------------------------------------------
# pmcprg's own conventions
# --------------------------------------------------------------------------

def test_pmcprg_rotation_and_transpose_conventions():
    """pmcprg's BB1 rotations, measured like the references' (pilot's test)."""
    pars = (1.2, 1.5)
    base, _, _ = _pair("BB1", "bb1", 0, pars)
    u, v = POINTS[:, 0], POINTS[:, 1]
    ref = {180: np.column_stack([1 - u, 1 - v]), 90: np.column_stack([1 - u, v]),
           270: np.column_stack([u, 1 - v])}
    for rotation, reflected in ref.items():
        for member in MEMBERS[("bb1", rotation)]:
            cop, tr, _ = _pair(member, "bb1", rotation, pars)
            # pmcprg never forms 1 − u (log1p kernels), the right-hand side
            # does: measured 1.3e-15.
            np.testing.assert_allclose(cop.pdf_array(POINTS), base.pdf_array(reflected), rtol=1e-14)
            # The transpose used for h2 has the transposed density (bit for
            # bit, measured: BB1's kernel is symmetric in its two coordinates).
            np.testing.assert_allclose(tr.pdf_array(POINTS[:, ::-1]), cop.pdf_array(POINTS),
                                       rtol=2e-15)
            assert np.sign(cop.params["tau_k"]) == (-1.0 if rotation in (90, 270) else 1.0)


# Relative error of the parameter the constructor recovers from τ (θ; δ for
# BB7), measured: BB1 2.0e-15 (θ = 2/(δ(1 − τ)) − 2 at θ = 0.3), BB6 5.3e-16
# (Joe's inversion), BB7 1.2e-15 (Brent on ln δ, xtol 1e-15), BB8 1.2e-14
# ((6, 0.4): Brent on ln(θ − 1) through the τ quadrature, xtol 1e-13).
ROUND_TRIP = {"bb1": 5e-15, "bb6": 2e-15, "bb7": 3e-15, "bb8": 5e-14}


@pytest.mark.parametrize("family, rotation", FAMILY_ROTATIONS, ids=FR_IDS)
def test_pmcprg_parameter_round_trip(family, rotation):
    for fam, rot, pars in CASE_KEYS:
        if (fam, rot) != (family, rotation):
            continue
        target = pars[1] if fam == "bb7" else pars[0]
        for member in MEMBERS[(fam, rot)]:
            built = _ours(member, fam, rot, pars)["built"]
            assert abs(built - target) <= ROUND_TRIP[fam] * target, (member, pars, built)


@pytest.mark.parametrize("family, rotation", FAMILY_ROTATIONS, ids=FR_IDS)
def test_scalar_paths_equal_array_paths(family, rotation):
    """pdf, cdf and inv_h: the scalar API is the array API bit for bit (measured)."""
    for fam, rot, pars in CASE_KEYS:
        if (fam, rot) != (family, rotation):
            continue
        for member in MEMBERS[(fam, rot)]:
            cop, _, _ = _pair(member, fam, rot, pars)
            ours = _ours(member, fam, rot, pars)
            np.testing.assert_array_equal([cop.pdf([a, b]) for a, b in POINTS], ours["pdf"])
            np.testing.assert_array_equal([cop.cdf([a, b]) for a, b in POINTS], ours["cdf"])
            np.testing.assert_array_equal([cop.inv_h(b, a) for a, b in POINTS], ours["hinv1"])


# --------------------------------------------------------------------------
# Parity: pmcprg against each package
# --------------------------------------------------------------------------

PARITY_PARAMS = [(n, f, r) for n in REFERENCES for f, r in FAMILY_ROTATIONS]
PARITY_IDS = [f"{n}-{f}{r}" for n, f, r in PARITY_PARAMS]


@pytest.mark.parametrize("reference, family, rotation", PARITY_PARAMS, ids=PARITY_IDS)
def test_interior_parity(reference, family, rotation):
    failures, compared = [], 0
    for key, case in _cases(reference).items():
        fam, rot, pars = key
        if (fam, rot) != (family, rotation):
            continue
        for member in MEMBERS[(fam, rot)]:
            ours = _ours(member, fam, rot, pars)
            for q in VECTOR_QUANTITIES:
                ref = _ref_vector(case, q)
                err = _errors(q, ours[q], ref, reference)
                tol = _tolerance(reference, fam, rot, pars, q)
                compared += err.size
                if not err.max() <= tol:
                    k = int(np.argmax(err))
                    failures.append(f"{member} {q} {pars} at {tuple(POINTS[k])}: pmcprg "
                                    f"{ours[q][k]!r}, {reference} {ref[k]!r} "
                                    f"(error {err[k]:.2e} > {tol:g})")
    assert compared > 0
    assert not failures, "\n".join(failures)


# pmcprg's inverse of its τ map at pyvinecopulib's τ — within 2.5e-16 of
# mpmath and of pmcprg's (≤ 4.2e-16), so a test of the inversion on another
# implementation's τ. Relative error of the recovered parameter, measured:
# 2.0e-15 (BB1, θ = 0.3), 0 (BB6), 1.2e-15 (BB7), 1.3e-14 (BB8, its Brent
# xtol).
THETA_AT_REF_TAU = {"bb1": 5e-15, "bb6": 2e-15, "bb7": 3e-15, "bb8": 5e-14}


@pytest.mark.parametrize("reference, family, rotation", PARITY_PARAMS, ids=PARITY_IDS)
def test_tau_and_tail_dependence_parity(reference, family, rotation):
    """τ(θ, δ), λ_L, λ_U — and pmcprg's parameter from pyvinecopulib's τ."""
    failures = []
    for key, case in _cases(reference).items():
        fam, rot, pars = key
        if (fam, rot) != (family, rotation):
            continue
        for member in MEMBERS[(fam, rot)]:
            ours = _ours(member, fam, rot, pars)
            for q in SCALAR_QUANTITIES:
                tol = _tolerance(reference, fam, rot, pars, q)
                if not abs(ours[q] - case[q]) <= tol:
                    failures.append(f"{member} {q} {pars}: pmcprg {ours[q]!r}, {reference} {case[q]!r}")
            if reference == "pyvinecopulib":
                assert _limited(reference, fam, pars, "tau") is None
                built = _recovered(_build(member, fam, rot, pars, tau=case["tau"]), fam)
                target = pars[1] if fam == "bb7" else pars[0]
                if not abs(built - target) <= THETA_AT_REF_TAU[fam] * target:
                    failures.append(f"{member} parameter at the package's τ {pars}: pmcprg {built!r}")
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("reference, family, rotation", PARITY_PARAMS, ids=PARITY_IDS)
def test_bivariate_law_conditions_on_either_margin(reference, family, rotation):
    """``BivariateLaw`` with uniform margins (the pilot's check): P(V ≤ b | U = a)
    is the package's h1(a, b), P(U ≤ a | V = b) its h2(a, b), both conditional
    densities its c(a, b) — through the tolerances of the copula itself."""
    from scipy import stats

    from pmcprg.copulas import BivariateLaw
    from pmcprg.numerics import EPS, ONE_MINUS_EPS

    failures, compared = [], 0
    for key, case in _cases(reference).items():
        fam, rot, pars = key
        if (fam, rot) != (family, rotation):
            continue
        for member in MEMBERS[(fam, rot)]:
            cop, _, _ = _pair(member, fam, rot, pars)
            law = BivariateLaw(cop, (stats.uniform,), (stats.uniform,))
            ours = {
                "h1": np.array([law.conditional_cdf(b, a, which="left") for a, b in POINTS]),
                "h2": np.array([law.conditional_cdf(a, b, which="right") for a, b in POINTS]),
                "pdf_left": np.array([law.conditional_pdf(b, a, which="left") for a, b in POINTS]),
                "pdf_right": np.array([law.conditional_pdf(a, b, which="right") for a, b in POINTS]),
            }
            for q, col in (("h1", "h1"), ("h2", "h2"), ("pdf_left", "pdf"), ("pdf_right", "pdf")):
                ref = _ref_vector(case, col)
                # The law floors the copula density at EPS (as the pilot):
                # BB1 (2.5, 3) has c = 5.7e-18 at (0.01, 0.99).
                keep = ref > EPS if col == "pdf" else np.ones(ref.size, bool)
                err = _errors(col, ours[q][keep], ref[keep], reference)
                tol = _tolerance(reference, fam, rot, pars, col)
                compared += err.size
                if not err.max() <= tol:
                    failures.append(f"{member} {q} {pars}: {err.max():.2e} > {tol:g}")
            # Sampling the left margin given the right one inverts h2: the
            # transpose's inv_h, as the law builds it from the parameters.
            law.set_seed(20260921)
            b = float(POINTS[3, 1])
            drawn = law.sample_conditional(b, which="right", n=8)
            ws = np.random.default_rng(20260921).uniform(EPS, ONE_MINUS_EPS, 8)
            np.testing.assert_array_equal(drawn, cop.transposed().inv_h_array(ws, np.full(8, b)))
    assert compared > 0
    assert not failures, "\n".join(failures)


# --------------------------------------------------------------------------
# The packages against one another
# --------------------------------------------------------------------------

# Largest mutual error of the two packages, outside REFERENCE_LIMITED (a
# cell is skipped when either package is limited on it).
REFERENCE_TOL = {
    # 2.6e-15 (BB1 (2.5, 3) at (0.99, 0.01)): the same formulas — each is
    # 1.8e-14 from mpmath at BB1 (0.3, 1.1)'s corners, together.
    "pdf": 5e-15,
    # 3.9e-15 (BB8 (3, 0.9) at (0.99, 0.99)).
    "cdf": 1e-14,
    # 3.2e-14 (BB7 (1.5, 5) at (0.01, 0.5): pyvinecopulib's h is the one off).
    "h1": 5e-14,
    "h2": 5e-14,
    # Every inverse is in the register.
    "hinv1": 5e-14,
    "hinv2": 5e-14,
    # Only BB1's τ is compared (VineCopula's others are in the register): exact.
    "tau": 5e-16,
    "lambda_L": 5e-16,
    "lambda_U": 5e-16,
}


def test_references_agree_with_each_other():
    a, b = REFERENCES
    failures, compared = [], 0
    for key in CASE_KEYS:
        fam, rot, pars = key
        ca, cb = _cases(a)[key], _cases(b)[key]
        for q in VECTOR_QUANTITIES + SCALAR_QUANTITIES:
            if _limited(a, fam, pars, q) or _limited(b, fam, pars, q):
                continue
            x = np.atleast_1d(np.array(ca[q], dtype=float))
            y = np.atleast_1d(np.array(cb[q], dtype=float))
            if q in H:
                x, y = np.clip(x, *VINECOPULA_H_RANGE), np.clip(y, *VINECOPULA_H_RANGE)
            err = np.abs(x - y) / (np.abs(y) if q == "pdf" else 1.0)
            compared += err.size
            if not err.max() <= REFERENCE_TOL[q]:
                failures.append(f"{fam}/{rot} {pars} {q}: {err.max():.2e}")
    assert compared > 0
    assert not failures, "\n".join(failures)


# --------------------------------------------------------------------------
# mpmath oracle: the textbook CDF and generator of each family
# --------------------------------------------------------------------------

def _mp_base_cdf(family: str, pars: tuple):
    """The textbook CDF of the unrotated family, Joe (1997), ch. 5 (module docstring)."""
    th, de = (mp.mpf(x) for x in pars)
    if family == "bb1":
        def C(u, v):
            return (1 + ((u ** -th - 1) ** de + (v ** -th - 1) ** de) ** (1 / de)) ** (-1 / th)
    elif family == "bb6":
        def C(u, v):
            p, q = -mp.log(1 - (1 - u) ** th), -mp.log(1 - (1 - v) ** th)
            return 1 - (1 - mp.exp(-(p ** de + q ** de) ** (1 / de))) ** (1 / th)
    elif family == "bb7":
        def C(u, v):
            a, b = 1 - (1 - u) ** th, 1 - (1 - v) ** th
            return 1 - (1 - (a ** -de + b ** -de - 1) ** (-1 / de)) ** (1 / th)
    elif family == "bb8":
        def C(u, v):
            eta = 1 - (1 - de) ** th
            A = (1 - (1 - de * u) ** th) * (1 - (1 - de * v) ** th) / eta
            return (1 - (1 - A) ** (1 / th)) / de
    else:
        raise KeyError(family)
    return C


def _mp_generator(family: str, pars: tuple):
    """(φ, φ′) of the textbook generator (Joe 1997, ch. 5), φ′ differentiated by hand."""
    th, de = (mp.mpf(x) for x in pars)
    if family == "bb1":                       # (t^−θ − 1)^δ
        return (lambda t: (t ** -th - 1) ** de,
                lambda t: -de * th * t ** (-th - 1) * (t ** -th - 1) ** (de - 1))
    if family == "bb6":                       # (−log(1 − (1 − t)^θ))^δ
        def dphi(t):
            m = 1 - (1 - t) ** th
            return -de * (-mp.log(m)) ** (de - 1) * th * (1 - t) ** (th - 1) / m
        return lambda t: (-mp.log(1 - (1 - t) ** th)) ** de, dphi
    if family == "bb7":                       # (1 − (1 − t)^θ)^−δ − 1
        return (lambda t: (1 - (1 - t) ** th) ** -de - 1,
                lambda t: -de * th * (1 - t) ** (th - 1) * (1 - (1 - t) ** th) ** (-de - 1))
    if family == "bb8":                       # −log((1 − (1 − δt)^θ)/(1 − (1 − δ)^θ))
        eta = 1 - (1 - de) ** th
        return (lambda t: -mp.log((1 - (1 - de * t) ** th) / eta),
                lambda t: -th * de * (1 - de * t) ** (th - 1) / (1 - (1 - de * t) ** th))
    raise KeyError(family)


def _mp_copula_cdf(family: str, rotation: int, pars: tuple):
    C = _mp_base_cdf(family, pars)
    return {0: C,
            90: lambda u, v: v - C(1 - u, v),
            180: lambda u, v: u + v - 1 + C(1 - u, 1 - v),
            270: lambda u, v: u - C(u, 1 - v)}[rotation]


def _mp_root(f, start: float):
    x0 = mp.mpf(start)
    root = mp.findroot(f, (x0, x0 * (1 + mp.mpf("1e-9"))), solver="secant")
    assert abs(f(root)) < mp.mpf(10) ** (-(mp.mp.dps - 10))
    return root


def _mp_value(family: str, rotation: int, pars: tuple, q: str, a: float, b: float, start: float):
    """q at (a, b) from the textbook CDF alone: derivatives by mp.diff, inverses by root finding.

    The density is a mixed difference of CDF values, which loses about
    log10(1/c) digits where it is small: the precision is raised by that
    much — ``start``, pmcprg's own value, only sets the precision — and the
    result is recomputed 15 digits higher and must agree
    (``test_parity_extra.py``).
    """
    C = _mp_copula_cdf(family, rotation, pars)
    u, v = mp.mpf(a), mp.mpf(b)

    def h1(x, y):
        return mp.diff(lambda s: C(s, y), x)

    def h2(x, y):
        return mp.diff(lambda s: C(x, s), y)

    if q == "cdf":
        return C(u, v)
    if q == "pdf":
        lost = max(0, int(-math.log10(max(abs(start), 1e-300))))
        values = []
        for extra in (0, 15):
            with mp.workdps(mp.mp.dps + lost + extra):
                values.append(mp.diff(C, (mp.mpf(a), mp.mpf(b)), (1, 1)))
        assert abs(values[0] - values[1]) <= abs(values[1]) * mp.mpf(10) ** (-(mp.mp.dps - 10))
        return values[1]
    if q == "h1":
        return h1(u, v)
    if q == "h2":
        return h2(u, v)
    if q == "hinv1":
        return _mp_root(lambda y: h1(u, y) - v, start)
    if q == "hinv2":
        return _mp_root(lambda x: h2(x, v) - u, start)
    raise KeyError(q)


def _mp_tau(family: str, rotation: int, pars: tuple):
    """Kendall's τ = 1 + 4 ∫₀¹ φ/φ′ (Genest & MacKay 1986) of the textbook
    generator, whatever closed form a package uses; −τ for 90°/270°.

    At 40 digits: at 20, the complements 1 − (1 − t)^θ near the ends cost
    BB6 (5, 1.5) 1.3·10⁻¹²; at 40 the result is within 2.2·10⁻²⁰ of the
    literature's closed forms on every case (module docstring).
    """
    phi, dphi = _mp_generator(family, pars)

    def ratio(t):
        # Within 10⁻⁴⁰ of an end, 1 − t or 1 − (1 − t)^θ rounds to 0 and
        # φ/φ′ is 0/0; its true value there is O(10⁻⁴⁰) (φ/φ′ → 0 at both
        # ends of a strict generator), so 0 is exact to the working precision.
        try:
            r = phi(t) / dphi(t)
        except ZeroDivisionError:
            return mp.mpf(0)
        return r if mp.isfinite(r) else mp.mpf(0)

    with mp.workdps(40):
        tau = 1 + 4 * mp.quad(ratio, [0, mp.mpf(1) / 2, 1])
    return -tau if rotation in (90, 270) else tau


def _mp_tail_dependence(family: str, rotation: int, pars: tuple):
    """(λ_L, λ_U) as the diagonal limits C(u, u)/u and (1 − 2u + C(u, u))/(1 − u)
    at u = x and 1 − x, x = 10⁻¹⁰⁰.

    The slowest of these ratios converges like x^θ (BB1's lower tail,
    θ = 0.3: 10⁻³⁰). The textbook CDFs form 1 − u, and (1 − u)^θ = 10⁻¹⁰⁰ᶿ
    next to 1: hence 100·max(θ, 1) + 100 digits.
    """
    C = _mp_copula_cdf(family, rotation, pars)
    with mp.workdps(int(100 * max(pars[0], 1)) + 100):
        lo = mp.mpf(10) ** -100
        hi = 1 - lo
        return C(lo, lo) / lo, (1 - 2 * hi + C(hi, hi)) / (1 - hi)


@pytest.mark.parametrize("family, rotation", FAMILY_ROTATIONS, ids=FR_IDS)
def test_mpmath_oracle_values(family, rotation):
    """pdf, cdf, h1, h2, hinv1, hinv2 of every member at every case and point, against mpmath (30 digits)."""
    pytest.importorskip("mpmath")
    failures = []
    for fam, rot, pars in CASE_KEYS:
        if (fam, rot) != (family, rotation):
            continue
        for member in MEMBERS[(fam, rot)]:
            ours = _ours(member, fam, rot, pars)
            for q in VECTOR_QUANTITIES:
                tol = _oracle_tolerance(fam, rot, q)
                for k, (a, b) in enumerate(POINTS):
                    with mp.workdps(30):
                        exact = _mp_value(fam, rot, pars, q, a, b, start=ours[q][k])
                        err = float(abs(ours[q][k] - exact) / (abs(exact) if q == "pdf" else 1))
                    if not err <= tol:
                        failures.append(f"{member} {q} {pars} at {(a, b)}: {err:.2e} > {tol:g}")
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("family, rotation", FAMILY_ROTATIONS, ids=FR_IDS)
def test_mpmath_oracle_tau_and_tail_dependence(family, rotation):
    """τ (Genest–MacKay, 40 digits), λ_L and λ_U (diagonal limits) of every member against mpmath."""
    pytest.importorskip("mpmath")
    failures = []
    for fam, rot, pars in CASE_KEYS:
        if (fam, rot) != (family, rotation):
            continue
        exact = {"tau": _mp_tau(fam, rot, pars)}
        exact["lambda_L"], exact["lambda_U"] = _mp_tail_dependence(fam, rot, pars)
        for member in MEMBERS[(fam, rot)]:
            ours = _ours(member, fam, rot, pars)
            for q, value in exact.items():
                if not float(abs(ours[q] - value)) <= _oracle_tolerance(fam, rot, q):
                    failures.append(f"{member} {q} {pars}: pmcprg {ours[q]!r}, mpmath {value}")
    assert not failures, "\n".join(failures)


def _worst(entry: Limited, q: str):
    """(error, key, point index) of the largest |pmcprg − package| in the entry's scope."""
    worst = (-1.0, None, None)
    for key, case in _cases(entry.reference).items():
        fam, rot, pars = key
        if not entry.covers(entry.reference, fam, pars, q):
            continue
        ours = _ours(MEMBERS[(fam, rot)][0], fam, rot, pars)
        if q in SCALAR_QUANTITIES:
            err, k = abs(ours[q] - case[q]), None
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
    ids=[f"{e.reference}-{e.family}-{'all' if e.pars is None else '-'.join(map(str, e.pars[0]))}-{q}"
         for e, q in LIMITED_PARAMS])
def test_reference_limited_entries_are_the_reference(entry, q):
    """At the worst point of each entry, mpmath sides with pmcprg, not with the package."""
    pytest.importorskip("mpmath")
    err, key, k = _worst(entry, q)
    assert key is not None
    fam, rot, pars = key
    # Still needed (else the entry must go), and within what the entry allows.
    assert _tolerance("none", fam, rot, pars, q) < err <= entry.tol, (err, entry)
    ours = _ours(MEMBERS[(fam, rot)][0], fam, rot, pars)
    with mp.workdps(40):
        if q == "tau":
            exact = _mp_tau(fam, rot, pars)
            ours_value, ref_value = ours["tau"], _cases(entry.reference)[key]["tau"]
        else:
            a, b = POINTS[k]
            exact = _mp_value(fam, rot, pars, q, a, b, start=ours[q][k])
            ours_value, ref_value = ours[q][k], _cases(entry.reference)[key][q][k]
        scale = abs(exact) if q == "pdf" else 1
        ours_err = float(abs(ours_value - exact) / scale)
        ref_err = float(abs(ref_value - exact) / scale)
    # pmcprg within its own tolerance of the exact value; the package carries
    # the discrepancy.
    assert ours_err <= _oracle_tolerance(fam, rot, q), (ours_err, entry)
    assert ref_err >= 0.9 * err, (ref_err, err, entry)


@pytest.mark.parametrize("cell", sorted(PMCPRG_LIMITED), ids=lambda c: f"{c[0]}{c[1]}-{c[2]}")
def test_pmcprg_limited_cells_are_pmcprg(cell):
    """Each PMCPRG_LIMITED cell is still needed and within what it allows, for every member."""
    pytest.importorskip("mpmath")
    family, rotation, q = cell
    worst = -1.0
    for fam, rot, pars in CASE_KEYS:
        if (fam, rot) != (family, rotation):
            continue
        for member in MEMBERS[(fam, rot)]:
            ours = _ours(member, fam, rot, pars)
            for k, (a, b) in enumerate(POINTS):
                with mp.workdps(30):
                    exact = _mp_value(fam, rot, pars, q, a, b, start=ours[q][k])
                    worst = max(worst, float(abs(ours[q][k] - exact)))
    assert worst <= PMCPRG_LIMITED[cell], (cell, worst)
    # "Still needed" is a last-bits statement, measured on macOS arm64: the
    # platform's libm moves these errors by an ulp or so (Hüsler–Reiss h⁻¹:
    # 2.56e-15 there, 1.78e-15 on the Linux CI runners), as for the goldens.
    if _BIT_EXACT:
        assert ORACLE_TOL[q] < worst, (cell, worst)
    # No package has an accurate inverse to compare with (REFERENCE_LIMITED).
    assert all(_limited(name, family, pars, q) for name in REFERENCES
               for f, r, pars in CASE_KEYS if (f, r) == (family, rotation))
