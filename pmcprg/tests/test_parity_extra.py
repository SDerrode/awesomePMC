"""Interior parity of the families beyond the pilot — audit FR-11, wave 1.

The extreme-value families (Galambos, Hüsler–Reiss, t-EV and the asymmetric
logistic Tawn model: ``TAWN1``, ``TAWN2``, ``TAWN3``), Plackett, AMH, FGM, and
Nelsen's Archimedean families 4.2.12 and 4.2.14 with their 90°/270°
rotations — 42 parameter sets on the 25 interior points of the pilot
(``scripts/parity/cases_extra.csv``, ``points.csv``) — against every package
that has them, and against ``mpmath`` everywhere:

* pyvinecopulib 1.0.0 — Tawn (three parameters, every quantity);
* VineCopula 2.6.1 — Tawn types 1 and 2 (families 104, 204; every quantity);
* R ``copula`` 1.1-7 — Galambos, Hüsler–Reiss, t-EV (pdf, cdf, τ, λ, A),
  Tawn through Khoudraji's device (pdf, cdf), Plackett, AMH (with h and
  h⁻¹), FGM;
* copBasic 2.2.16 — the closed-form CDFs (A12 and its rotations included);
* fCopulae 4052.86 — Galambos, Hüsler–Reiss, Tawn (three parameters), AMH,
  A12, A14 (pdf, cdf, τ; λ and A for the EV families).

The tables are generated offline by ``scripts/parity/gen_r_extra.R`` and
``scripts/parity/gen_pyvinecopulib_extra.py`` (``scripts/parity/README.md``);
nothing here needs R or pyvinecopulib.

No quantity rests on one implementation: the **mpmath oracle**
(:func:`test_mpmath_oracle_values`, :func:`test_mpmath_oracle_tau_tail_pickands`)
evaluates every quantity of every case on every point from the textbook CDF
alone — pdf and h by ``mp.diff``, h⁻¹ by root finding, τ by the Genest–MacKay
or Hoeffding integrals, λ by the diagonal limits or 2 − 2A(½) — so the h and
h⁻¹ that no package gives (the EV families, Plackett, FGM, A12, A14 and all
rotations) and the quantities one package alone gives (A14, Tawn's A, t-EV's
τ and λ) are checked all the same.

Convention mapping (each measured by the generators, re-read by
:func:`test_recorded_conventions`)
------------------------------------------------------------------
* **Key.** galambos [θ], husler_reiss [λ], tev [ρ, ν], tawn [θ, ψ_u, ψ_v]
  with ψ_u the weight of −log u, plackett/amh/fgm/a12/a14 [θ] of the
  unrotated family; the textbook CDFs are the ``_mp_base_cdf`` below.
  Every package's parametrisation is the key's (its CDF equals the textbook
  one to ≤ 4.4·10⁻¹⁵, the median to ≤ 6·10⁻¹⁷) with these native maps: copula's
  ``huslerReissCopula(δ = λ)``, ``tevCopula(ρ, df = ν)``,
  ``khoudrajiCopula(indep, gumbel(θ), shapes = (ψ_u, ψ_v))``; copBasic's
  ``khoudrajiPCOP(alpha = 1 − ψ_u, beta = 1 − ψ_v)``; fCopulae's Tawn
  ``param = (α, β, r) = (ψ_u, ψ_v, θ)``; pyvinecopulib's
  ``[psi1, psi2, theta] = [ψ_u, ψ_v, θ]``.
* **VineCopula's Tawn numbering is the other way round from pmcprg's.**
  Family **104** frees ψ_u (ψ_v = 1): pmcprg's ``TAWN2``; family **204**
  frees ψ_v (ψ_u = 1): pmcprg's ``TAWN1``
  (:func:`test_tawn_type_numbering_against_vinecopula`). The two types are
  transposes of each other, so the numerics of neither package are at fault;
  ``tawn.py``'s docstring, which claimed VineCopula's assignment from
  recollection, is corrected.
* **Rotations.** copBasic's ``reflect = "acute"`` is 90° (v − C(1 − u, v)),
  ``"grave"`` 270° (u − C(u, 1 − v)), pmcprg's convention (pilot).
* **A(t)** in the u-share convention t = log u / log(uv), pmcprg's and
  fCopulae's; copula's ``A()`` is in the v-share one and is recorded at
  1 − t (its three EV families are exchangeable anyway).
* **h-functions** as in the pilot: h1 = ``conditional_cdf(v, u)``, h2 the
  transposed copula's ``conditional_cdf(u, v)``; ``TAWN1``ᵀ = ``TAWN2``,
  ``TAWN3``ᵀ swaps the weights, 90°ᵀ = 270°.
* **pmcprg at the reference's parameter.** pmcprg is built from τ by its own
  τ(θ); its θ(τ) is tested on its own (:func:`test_pmcprg_theta_round_trip`:
  a Brent search for Galambos, 8·10⁻¹⁴, and a table interpolation for
  Plackett, 1.5·10⁻⁴), and the values are then evaluated at the key's θ,
  assigned to the built copula — they test the formulas, not the inversion.

Tolerances and discrepancies
----------------------------
:data:`TOL` (pmcprg against a package) and :data:`ORACLE_TOL` (pmcprg against
mpmath) are small multiples of the largest error measured on the grid,
stated next to each. Outside the two registers below, pmcprg agrees with
mpmath to 4.0·10⁻¹⁴ (pdf, relative), 2.3·10⁻¹⁶ (cdf), 2.7·10⁻¹⁵ (h),
5.3·10⁻¹⁶ (h⁻¹), 2.1·10⁻¹⁴ (τ), 4.4·10⁻¹⁶ (λ), 5.3·10⁻¹⁶ (A, relative).

* :data:`PMCPRG_LIMITED` — pmcprg is the one off: the h⁻¹ of the seven
  families that use :meth:`CopulaVirt.inv_h`'s generic Brent search, whose
  ``xtol = 1e-8`` leaves up to 2.5·10⁻⁹ (Galambos, Hüsler–Reiss, Plackett,
  AMH, FGM, A12, A14; copula's AMH inverse is 7·10⁻¹⁵ from mpmath); and the
  τ of Galambos and Hüsler–Reiss, whose ``scipy.integrate.quad`` of the
  Genest–MacKay integrand is 1.8·10⁻¹² off (both within their modules'
  tested accuracy). Not defects — each within what the code promises — but
  measured margins.
* :data:`REFERENCE_LIMITED` — a package is off and pmcprg agrees with mpmath,
  each entry proven at its worst point
  (:func:`test_reference_limited_entries_are_the_reference`).

The packages' mutual agreement outside these entries is asserted by
:func:`test_references_agree_with_each_other`.
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

from pmcprg.copulas import CopulaEnum
from pmcprg.copulas.archimedean.amh import _amh_tau_from_theta
from pmcprg.copulas.explicit.plackett import _plackett_tau_from_theta
from pmcprg.copulas.extreme_value import t_ev, tawn
from pmcprg.copulas.extreme_value.galambos import _galambos_A_terms, _galambos_tau_from_theta
from pmcprg.copulas.extreme_value.husler_reiss import (_husler_reiss_A_terms,
                                                       _husler_reiss_tau_from_lambda)

DATA = Path(__file__).resolve().parent / "data" / "parity"
REFERENCES = ("pyvinecopulib", "vinecopula", "rcopula", "copbasic", "fcopulae")
VECTOR_QUANTITIES = ("pdf", "cdf", "h1", "h2", "hinv1", "hinv2")
SCALAR_QUANTITIES = ("tau", "lambda_L", "lambda_U")
EV_FAMILIES = ("galambos", "husler_reiss", "tev", "tawn")
HINV = ("hinv1", "hinv2")


@functools.cache
def _table(name: str) -> dict:
    return json.loads((DATA / f"extra_{name}.json").read_text())


@functools.cache
def _cases(name: str) -> dict:
    return {(c["family"], c["rotation"], tuple(c["pars"])): c for c in _table(name)["cases"]}


POINTS = np.array(_table("rcopula")["points"], dtype=float)
TGRID = np.array(_table("rcopula")["pickands_points"], dtype=float)
# The whole grid, covered by a package or not (the A14 rotations by none).
CASE_KEYS = tuple((f, r, tuple(p)) for f, r, p in _table("rcopula")["grid"])
FAMILY_ROTATIONS = tuple(dict.fromkeys((f, r) for f, r, _ in CASE_KEYS))
FR_IDS = [f"{f}{r}" for f, r in FAMILY_ROTATIONS]

# VineCopula returns its h-functions clamped to this interval (pilot).
VINECOPULA_H_RANGE = (1e-12, 1.0 - 1e-12)

_BASE_MEMBER = {"galambos": "GALAMBOS", "husler_reiss": "HUSLER_REISS", "tev": "TEV",
                "plackett": "PLACKETT", "amh": "AMH", "fgm": "FGM", "a12": "A12", "a14": "A14"}
TRANSPOSE = {"TAWN1": "TAWN2", "TAWN2": "TAWN1", "A1290": "A12270", "A12270": "A1290",
             "A1490": "A14270", "A14270": "A1490"}


def _members(family: str, rotation: int, pars: tuple) -> tuple:
    """The pmcprg members a case is compared with: a Tawn case with one weight
    1 is both its two-parameter type and ``TAWN3``."""
    if family == "tawn":
        _, pu, pv = pars
        two = ("TAWN1",) if pu == 1.0 else ("TAWN2",) if pv == 1.0 else ()
        return two + ("TAWN3",)
    base = _BASE_MEMBER[family]
    return (base if rotation == 0 else f"{base}{rotation}",)


# --------------------------------------------------------------------------
# pmcprg side
# --------------------------------------------------------------------------

def _tev_s(rho: float) -> float:
    """t-EV's internal parameter s = ln((1 − ρ)/(1 + ρ)) (``t_ev.py``)."""
    return math.log1p(-rho) - math.log1p(rho)


def _tau_of_theta(family: str, pars: tuple) -> float:
    """pmcprg's own τ of the unrotated family at the key's parameters."""
    th = pars[0]
    if family == "galambos":
        return _galambos_tau_from_theta(th)
    if family == "husler_reiss":
        return _husler_reiss_tau_from_lambda(th)
    if family == "tev":
        return t_ev._tau_of(_tev_s(th), pars[1])
    if family == "tawn":
        return tawn._tau_of(*pars)
    if family == "plackett":
        return _plackett_tau_from_theta(th)
    if family == "amh":
        return _amh_tau_from_theta(th)
    if family == "fgm":
        return 2.0 * th / 9.0
    if family == "a12":
        return 1.0 - 2.0 / (3.0 * th)
    if family == "a14":
        return 1.0 - 2.0 / (1.0 + 2.0 * th)
    raise KeyError(family)


def _build(member: str, family: str, rotation: int, pars: tuple, tau: float | None = None):
    """pmcprg's copula at τ (by default pmcprg's τ of the key's parameters)."""
    if tau is None:
        tau = _tau_of_theta(family, pars) * (-1.0 if rotation in (90, 270) else 1.0)
    kwargs = {"tau_k": tau}
    if family == "tev":
        kwargs["nu"] = pars[1]
    if member == "TAWN1":
        kwargs["psi"] = pars[2]
    elif member == "TAWN2":
        kwargs["psi"] = pars[1]
    elif member == "TAWN3":
        kwargs.update(psi_u=pars[1], psi_v=pars[2])
    return getattr(CopulaEnum, member).klass(**kwargs)


def _theta(cop) -> float:
    """The dependence parameter pmcprg built (ρ for t-EV; the base's θ for a rotation)."""
    return getattr(cop, "_base", cop).theta


def _pin(cop, theta: float) -> None:
    """Evaluate at exactly the key's θ (module docstring): every path of these
    families reads ``theta`` (a rotation: its base's)."""
    base = getattr(cop, "_base", None)
    if base is not None:
        base.theta = theta
    cop.theta = theta


def _pickands(family: str, pars: tuple):
    if family == "galambos":
        return _galambos_A_terms(TGRID, pars[0])[0]
    if family == "husler_reiss":
        return _husler_reiss_A_terms(TGRID, pars[0])[0]
    if family == "tev":
        return t_ev._tev_A_terms(TGRID, _tev_s(pars[0]), pars[1])[0]
    if family == "tawn":
        return tawn._tawn_A_terms(TGRID, *pars)[0]
    return None


@functools.cache
def _pair(member: str, family: str, rotation: int, pars: tuple):
    """(copula, its ``transposed()``, the θ its constructor built), at the key's θ."""
    cop = _build(member, family, rotation, pars)
    built = _theta(cop)
    tr = cop.transposed()
    assert type(tr) is getattr(CopulaEnum, TRANSPOSE.get(member, member)).klass
    if family != "tev":          # t-EV: its memo returns the very s of the key
        _pin(cop, pars[0])
        if tr is not cop:
            _pin(tr, pars[0])
    return cop, tr, built


@functools.cache
def _ours(member: str, family: str, rotation: int, pars: tuple) -> dict:
    cop, tr, built = _pair(member, family, rotation, pars)
    u, v = POINTS[:, 0].copy(), POINTS[:, 1].copy()
    lam_l, lam_u = cop.tail_dependence()
    return {
        "theta_built": built,
        "pdf": cop.pdf_array(POINTS),
        "cdf": cop.cdf_array(POINTS),
        "h1": np.array([cop.conditional_cdf(b, a) for a, b in POINTS]),
        "h2": np.array([tr.conditional_cdf(a, b) for a, b in POINTS]),
        "hinv1": cop.inv_h_array(v, u),
        "hinv2": tr.inv_h_array(u, v),
        "tau": float(cop.params["tau_k"]),
        "lambda_L": lam_l,
        "lambda_U": lam_u,
        "A": _pickands(family, pars),
    }


def _ref_vector(case: dict, q: str) -> np.ndarray | None:
    values = case.get(q)
    if values is None:
        return None
    assert None not in values, f"partial null in {case['native']} {q}"
    return np.array(values, dtype=float)


def _errors(q: str, x, ref, reference: str) -> np.ndarray:
    """|x − ref|, relative for the density and A; VineCopula's h compared clamped."""
    x = np.asarray(x, dtype=float)
    if reference == "vinecopula" and q in ("h1", "h2"):
        x = np.clip(x, *VINECOPULA_H_RANGE)
    d = np.abs(x - ref)
    return d / np.abs(ref) if q in ("pdf", "A") else d


# --------------------------------------------------------------------------
# Tolerances (module docstring). Absolute; the density's and A's relative.
# --------------------------------------------------------------------------

# pmcprg against a package, outside the registers below.
TOL = {
    # 4.6e-14: fCopulae's A12, θ = 8 at (0.99, 0.5) (fCopulae itself 3.9e-14
    # from mpmath); 3.4e-14 against copula's t-EV (copula 3.4e-14 off).
    "pdf": 1e-13,
    # 4.4e-15: AMH θ = 0.95 at (0.99, 0.99) against copula and fCopulae, whose
    # textbook uv/(1 − θ(1 − u)(1 − v)) is 4.35e-15 from mpmath (pmcprg 9e-17).
    "cdf": 5e-15,
    # 6.6e-15: copula's AMH at (0.99, 0.99) (its own error, 6.5e-15).
    "h1": 1e-14,
    "h2": 1e-14,
    # No accurate package inverse outside the registers (the Tawn inverses
    # of both vine libraries are numerical); copula's AMH inverse is 7.3e-15
    # from mpmath.
    "hinv1": 1e-14,
    "hinv2": 1e-14,
    # 2.1e-14: pyvinecopulib's Tawn (1.5, 1, 0.6) — pmcprg's quadrature
    # (pyvinecopulib is exact there).
    "tau": 5e-14,
    # 2.2e-16 (Tawn against pyvinecopulib and fCopulae).
    "lambda_L": 1e-15,
    "lambda_U": 1e-15,
    # 5.6e-16: fCopulae's Tawn at t = 0.05 (pmcprg ≤ 5.3e-16 from mpmath).
    "A": 1e-15,
}

# pmcprg against mpmath, outside PMCPRG_LIMITED.
ORACLE_TOL = {
    # 4.0e-14: Hüsler–Reiss λ = 5 at (0.01, 0.99), c = 5.7e-51; 1.4e-14 for
    # A12/A14 θ = 8.
    "pdf": 1e-13,
    # 2.3e-16 (Plackett θ = 25 at (0.99, 0.99)).
    "cdf": 1e-15,
    # 2.7e-15 (A14 θ = 8).
    "h1": 1e-14,
    "h2": 1e-14,
    # 5.3e-16 (Tawn, bisection on logit v).
    "hinv1": 2e-15,
    "hinv2": 2e-15,
    # 2.1e-14 (Tawn (1.5, 1, 0.6), graded Gauss–Legendre rule).
    "tau": 5e-14,
    # 4.4e-16 (t-EV, Tawn).
    "lambda_L": 1e-15,
    "lambda_U": 1e-15,
    # 5.3e-16 (Tawn).
    "A": 1e-15,
}

# pmcprg is the one off (module docstring), measured against mpmath:
# (family, rotation, quantity) → tolerance.
PMCPRG_LIMITED = {
    # CopulaVirt.inv_h's generic brentq, xtol = 1e-8 (the rotations of A12
    # and A14 have their own monotone Newton inverse, 3.9e-16).
    **{(f, 0, q): 5e-9 for f in ("galambos", "husler_reiss", "plackett", "amh", "fgm", "a12", "a14")
       for q in HINV},
    # scipy.integrate.quad of the Genest–MacKay integrand (_pickands.tau_from_A_terms):
    # 1.8e-12 at θ = 0.3 (Galambos), 1.1e-12 at λ = 1.5 (Hüsler–Reiss).
    ("galambos", 0, "tau"): 5e-12,
    ("husler_reiss", 0, "tau"): 5e-12,
}
# Measured maxima of the h⁻¹ entries: 2.5e-9 (A12, A14, Galambos, Plackett),
# 2.4e-9 (Hüsler–Reiss), 2.3e-9 (AMH, FGM).


@dataclass(frozen=True)
class Limited:
    """A cell where the *package* is off — its tolerance is the package's error.

    ``pars`` None: every parameter set of the family. Proven by
    :func:`test_reference_limited_entries_are_the_reference` with mpmath at
    its worst point, and must still be needed (worst error above the default).
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


REFERENCE_LIMITED = (
    Limited("pyvinecopulib", "tawn", None, ("pdf",), 5e-10,
            "vinecopulib's Tawn density in the far corner: 2.0e-10 relative at (0.99, 0.01), "
            "(θ, ψ_u, ψ_v) = (4, 1, 0.3), c = 6.4e-7"),
    Limited("pyvinecopulib", "tawn", None, HINV, 6e-11,
            "numerical inversion: 2.9e-11"),
    Limited("vinecopula", "tawn", None, ("pdf",), 6e-10,
            "the same corner: 2.9e-10 relative"),
    Limited("vinecopula", "tawn", None, HINV, 1.2e-11,
            "numerical inversion: 5.7e-12"),
    Limited("vinecopula", "tawn", None, ("tau",), 2e-7,
            "numerical integration: 9.7e-8 at (1.5, 1, 0.6)"),
    Limited("fcopulae", "tawn", None, ("pdf",), 4e-10,
            "fCopulae's density from A, A', A'' in powers: 1.7e-10 relative at (0.99, 0.01)"),
    Limited("fcopulae", "tawn", None, ("tau",), 2e-7,
            "evTau: integrate() at its default tolerance, 9.7e-8 at (1.5, 1, 0.6)"),
    Limited("rcopula", "galambos", ((5.0,),), ("pdf",), 0.2,
            "copula's symbolic Galambos density cancels in the corner: 9.1e-2 relative at "
            "(0.01, 0.99), θ = 5, where c = 1.25e-13"),
    Limited("rcopula", "galambos", ((1.5,),), ("pdf",), 1e-10,
            "the same, 4.8e-11 relative at (0.01, 0.99), θ = 1.5"),
    Limited("fcopulae", "galambos", ((5.0,),), ("pdf",), 1e-2,
            "fCopulae's density from A, A', A'': 3.9e-3 relative at (0.01, 0.99), θ = 5"),
    Limited("fcopulae", "galambos", ((1.5,),), ("pdf",), 5e-12,
            "the same, 2.1e-12 relative at θ = 1.5"),
    Limited("fcopulae", "husler_reiss", ((5.0,),), ("pdf",), 2.0,
            "fCopulae's density cancels to −1.1e-16 at (0.99, 0.01), λ = 5, where "
            "c = 5.7e-51: relative error 1 against pmcprg (1.96e34 against mpmath)"),
    Limited("fcopulae", "husler_reiss", ((1.5,),), ("pdf",), 3e-12,
            "the same, 1.4e-12 relative at λ = 1.5"),
    Limited("rcopula", "galambos", None, ("tau",), 3e-8,
            "galambosTauFun, a spline of τ(θ): 1.4e-8 at θ = 1.5"),
    Limited("fcopulae", "galambos", None, ("tau",), 3e-8,
            "evTau, integrate() of A'': 1.4e-8 at θ = 1.5"),
    Limited("rcopula", "husler_reiss", None, ("tau",), 5e-8,
            "huslerReissTauFun, a spline: 2.3e-8 at λ = 1.5"),
    Limited("fcopulae", "husler_reiss", None, ("tau",), 5e-8,
            "evTau, integrate(): 2.3e-8 at λ = 1.5"),
    Limited("rcopula", "tev", ((0.5, 4.0),), ("tau",), 2e-7,
            "tevTauFun, a spline: 8.3e-8 at (ρ, ν) = (0.5, 4)"),
    Limited("rcopula", "tev", ((-0.3, 2.0), (0.8, 10.0)), ("tau",), 0.5,
            "tevTauFun ignores ν and the sign of ρ — a spline in ρ² of the ν = 4 curve: "
            "0.1225 at (−0.3, 2), which is τ(0.3, 4), for τ = 0.0720 "
            "(test_copula_tev_tau_is_the_nu4_curve_of_rho_squared)"),
    Limited("rcopula", "plackett", None, ("tau",), 1e-3,
            "plackettTauFun, a spline: 6.7e-4 at θ = 0.15"),
    Limited("rcopula", "plackett", ((25.0,),), ("pdf",), 5e-13,
            "copula's Plackett density at (0.99, 0.99), θ = 25: 2.2e-13 relative"),
    Limited("fcopulae", "a14", ((8.0,),), ("pdf",), 6e-13,
            "fCopulae's generic Archimedean density ψ''/(ψ'ψ'): 2.9e-13 relative, θ = 8"),
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


# --------------------------------------------------------------------------
# The tables
# --------------------------------------------------------------------------

def test_tables_share_grid_and_carry_provenance():
    root = Path(__file__).resolve().parents[2] / "scripts" / "parity"
    points = np.loadtxt(root / "points.csv", delimiter=",", skiprows=1)
    tgrid = np.loadtxt(root / "pickands_points.csv", skiprows=1)
    np.testing.assert_array_equal(POINTS, points)
    np.testing.assert_array_equal(TGRID, tgrid)
    assert POINTS.min() >= 0.01 and POINTS.max() <= 0.99          # interior only
    with open(root / "cases_extra.csv") as fh:
        assert len(fh.read().split()) - 1 == len(CASE_KEYS)
    covered = set()
    for name in REFERENCES:
        t = _table(name)
        np.testing.assert_array_equal(np.array(t["points"], dtype=float), POINTS)
        np.testing.assert_array_equal(np.array(t["pickands_points"], dtype=float), TGRID)
        assert tuple((f, r, tuple(p)) for f, r, p in t["grid"]) == CASE_KEYS
        assert set(_cases(name)) <= set(CASE_KEYS)
        covered |= set(_cases(name))
        for key in ("reference", "environment", "generated", "repository_version",
                    "command", "definitions", "convention_checks"):
            assert t[key], (name, key)
        assert t["reference"]["version"]
    # Every case but the A14 rotations has a package (those: mpmath only).
    assert set(CASE_KEYS) - covered == {k for k in CASE_KEYS if k[0] == "a14" and k[1]}


@pytest.mark.parametrize("name", REFERENCES)
def test_recorded_conventions(name):
    """The parametrisations of the module docstring are the ones the generators measured."""
    checks = _table(name)["convention_checks"]
    for key, fam_checks in checks["per_family"].items():
        for check, value in fam_checks.items():
            if not check.endswith("_median"):
                continue
            if check.startswith(("param_", "A_")):
                tol = checks["param_tolerance"]
            elif "hinv" in check:
                tol = checks["roundtrip_tolerance"]
            else:
                tol = checks["fd_tolerance"]
            assert value < tol, (name, key, check, value)
        if "param_cdf_abs_error_max" in fam_checks:
            # The textbook CDF to rounding: measured ≤ 4.4e-15 (AMH near (1, 1)).
            assert fam_checks["param_cdf_abs_error_max"] < 1e-14, (name, key)
    if name == "vinecopula":
        assert checks["tawn_codes"] == {"104": "psi_u", "204": "psi_v"}
    if name == "pyvinecopulib":
        assert checks["tawn_weights"] == "psi1 = psi_u, psi2 = psi_v"
    if name == "fcopulae":
        # fCopulae's default Archimedean density for AMH (type 3) is wrong —
        # .invPhiFirstDer/.invPhiSecondDer write e^y − 1 for e^y − θ — which is
        # why the table holds its per-type formula (alternative = TRUE).
        assert checks["fcopulae_amh_default_density_rel_error_max"] > 0.5


def test_tawn_type_numbering_against_vinecopula():
    """pmcprg's ``TAWN1`` (ψ_u = 1) is VineCopula's family 204, ``TAWN2`` its 104."""
    for (family, _, pars), case in _cases("vinecopula").items():
        assert family == "tawn"
        _, pu, pv = pars
        code = int(case["native"].split(",")[0].split("=")[1])
        assert (code, _members("tawn", 0, pars)[0]) in ((204, "TAWN1"), (104, "TAWN2"))
        assert (code == 204) == (pu == 1.0) and (code == 104) == (pv == 1.0)


# --------------------------------------------------------------------------
# pmcprg's own conventions
# --------------------------------------------------------------------------

# Relative error of θ(τ(θ)) (module docstring), measured: Galambos 8.1e-14
# (Brent, xtol = 1e-12), Hüsler–Reiss 1.6e-14, Plackett 1.5e-4 (linear
# interpolation of log θ in its τ table, plackett.py), AMH 1.2e-15, A12
# 4.4e-16, A14 2.2e-16, t-EV 1.1e-16 (ρ = −tanh(s/2)), FGM 0; Tawn exactly
# (its memo returns the θ that produced τ).
ROUND_TRIP = {"galambos": 3e-13, "husler_reiss": 5e-14, "plackett": 3e-4, "amh": 3e-15,
              "a12": 1e-15, "a14": 1e-15, "tev": 5e-16, "fgm": 0.0, "tawn": 0.0}


@pytest.mark.parametrize("family, rotation", FAMILY_ROTATIONS, ids=FR_IDS)
def test_pmcprg_theta_round_trip(family, rotation):
    for fam, rot, pars in CASE_KEYS:
        if (fam, rot) != (family, rotation):
            continue
        for member in _members(fam, rot, pars):
            built = _ours(member, fam, rot, pars)["theta_built"]
            assert abs(built - pars[0]) <= ROUND_TRIP[fam] * abs(pars[0]), (member, pars, built)


@pytest.mark.parametrize("family, rotation", FAMILY_ROTATIONS, ids=FR_IDS)
def test_scalar_paths_equal_array_paths(family, rotation):
    """pdf, cdf and inv_h: the scalar API is the array API bit for bit (measured),
    but AMH's density — N/W³ against exp(log N − 3 log W), 4.8e-16 relative."""
    rtol = 1e-15 if family == "amh" else 0.0
    for fam, rot, pars in CASE_KEYS:
        if (fam, rot) != (family, rotation):
            continue
        for member in _members(fam, rot, pars):
            cop, _, _ = _pair(member, fam, rot, pars)
            ours = _ours(member, fam, rot, pars)
            np.testing.assert_allclose([cop.pdf([a, b]) for a, b in POINTS], ours["pdf"],
                                       rtol=rtol, atol=0.0)
            np.testing.assert_array_equal([cop.cdf([a, b]) for a, b in POINTS], ours["cdf"])
            np.testing.assert_array_equal([cop.inv_h(b, a) for a, b in POINTS], ours["hinv1"])


# --------------------------------------------------------------------------
# Parity: pmcprg against each package
# --------------------------------------------------------------------------

PARITY_PARAMS = [(n, f, r) for n in REFERENCES for f, r in FAMILY_ROTATIONS
                 if any(k[:2] == (f, r) for k in _cases(n))]
PARITY_IDS = [f"{n}-{f}{r}" for n, f, r in PARITY_PARAMS]


@pytest.mark.parametrize("reference, family, rotation", PARITY_PARAMS, ids=PARITY_IDS)
def test_interior_parity(reference, family, rotation):
    failures, compared = [], 0
    for key, case in _cases(reference).items():
        fam, rot, pars = key
        if (fam, rot) != (family, rotation):
            continue
        for member in _members(*key):
            ours = _ours(member, fam, rot, pars)
            for q in VECTOR_QUANTITIES + ("A",):
                ref = _ref_vector(case, q)
                if ref is None:
                    continue
                err = _errors(q, ours[q], ref, reference)
                tol = _tolerance(reference, fam, rot, pars, q)
                compared += err.size
                if not err.max() <= tol:
                    k = int(np.argmax(err))
                    failures.append(f"{member} {q} {pars} at index {k}: pmcprg {ours[q][k]!r}, "
                                    f"{reference} {ref[k]!r} (error {err[k]:.2e} > {tol:g})")
    assert compared > 0
    assert not failures, "\n".join(failures)


# pmcprg's θ(τ) at a package's τ where that τ is a closed form (or, for
# pyvinecopulib's Tawn, within 1e-16 of mpmath): relative error measured
# 1.2e-15 (AMH, copula's series), 0 (FGM), 4.4e-16 (A12), 2.2e-16 (A14),
# 4.9e-14 (Tawn: Brent on ln(θ − 1), xtol 1e-13). fCopulae's AMH τ is an
# integrate() 4.2e-15 off, which θ(τ) amplifies to 2.9e-14: not used.
THETA_AT_REF_TAU = {("rcopula", "amh"): 5e-15, ("rcopula", "fgm"): 5e-16,
                    ("fcopulae", "a12"): 1e-15, ("fcopulae", "a14"): 1e-15,
                    ("pyvinecopulib", "tawn"): 1e-13}


@pytest.mark.parametrize("reference, family, rotation", PARITY_PARAMS, ids=PARITY_IDS)
def test_tau_and_tail_dependence_parity(reference, family, rotation):
    """τ(θ), λ_L, λ_U — and pmcprg's θ(τ) at the package's τ where it is accurate."""
    failures = []
    for key, case in _cases(reference).items():
        fam, rot, pars = key
        if (fam, rot) != (family, rotation):
            continue
        for member in _members(*key):
            ours = _ours(member, fam, rot, pars)
            for q in SCALAR_QUANTITIES:
                if case[q] is None:
                    continue
                tol = _tolerance(reference, fam, rot, pars, q)
                if not abs(ours[q] - case[q]) <= tol:
                    failures.append(f"{member} {q} {pars}: pmcprg {ours[q]!r}, {reference} {case[q]!r}")
            tol = THETA_AT_REF_TAU.get((reference, fam))
            if tol is not None:
                assert _limited(reference, fam, pars, "tau") is None
                tau = case["tau"] * (-1.0 if rot in (90, 270) else 1.0)
                built = _theta(_build(member, fam, rot, pars, tau=tau))
                if not abs(built - pars[0]) <= tol * abs(pars[0]):
                    failures.append(f"{member} θ(τ_ref) {pars}: pmcprg {built!r}")
    assert not failures, "\n".join(failures)


BIVARIATE_PARAMS = [p for p in PARITY_PARAMS if p[0] in ("pyvinecopulib", "vinecopula")]


@pytest.mark.parametrize("reference, family, rotation", BIVARIATE_PARAMS,
                         ids=[f"{n}-{f}{r}" for n, f, r in BIVARIATE_PARAMS])
def test_bivariate_law_conditions_on_either_margin(reference, family, rotation):
    """``BivariateLaw`` with uniform margins on the Tawn models (the pilot's check):
    P(V ≤ b | U = a) is the package's h1(a, b), P(U ≤ a | V = b) its h2(a, b),
    both conditional densities its c(a, b) — through the tolerances of the
    copula itself (measured: the maxima of :func:`test_interior_parity`)."""
    from scipy import stats

    from pmcprg.copulas import BivariateLaw
    from pmcprg.numerics import EPS, ONE_MINUS_EPS

    failures, compared = [], 0
    for key, case in _cases(reference).items():
        fam, rot, pars = key
        for member in _members(*key):
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
                keep = ref > EPS if col == "pdf" else np.ones(ref.size, bool)
                err = _errors(col, ours[q][keep], ref[keep], reference)
                tol = _tolerance(reference, fam, rot, pars, col)
                compared += err.size
                if not err.max() <= tol:
                    failures.append(f"{member} {q} {pars}: {err.max():.2e} > {tol:g}")
            # Sampling the left margin given the right one inverts h2: the
            # transpose's inv_h — as the law builds it, from the parameters
            # (its θ is not the one assigned to ``tr``: the Tawn transpose
            # rebuilds θ from (τ, ψ), 2.4e-15 away).
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

# Largest mutual error of two packages, outside REFERENCE_LIMITED (a cell is
# skipped when either package is limited on it).
REFERENCE_TOL = {
    # 2.1e-14: copula against fCopulae (AMH θ = 0.95 at (0.01, 0.01)).
    "pdf": 5e-14,
    # 4.4e-15: copula against copBasic (AMH θ = 0.95 at (0.99, 0.99): copula's
    # rounding, 4.35e-15 from mpmath, copBasic's 9e-17).
    "cdf": 1e-14,
    # 1.3e-15: pyvinecopulib against VineCopula (Tawn).
    "h1": 5e-15,
    "h2": 5e-15,
    # No pair outside the register (the Tawn inverses of both vine
    # libraries are numerical; copula alone has AMH's).
    "hinv1": 1e-14,
    "hinv2": 1e-14,
    # 4.2e-15: copula against fCopulae (AMH θ = −0.9: fCopulae's integrate()).
    "tau": 1e-14,
    # 2.2e-16 (Tawn).
    "lambda_L": 1e-15,
    "lambda_U": 1e-15,
    # 1.3e-16 (copula against fCopulae, Hüsler–Reiss).
    "A": 5e-16,
}
PAIRS = [(a, b) for i, a in enumerate(REFERENCES) for b in REFERENCES[i + 1:]
         if set(_cases(a)) & set(_cases(b))]


@pytest.mark.parametrize("a, b", PAIRS, ids=[f"{a}-{b}" for a, b in PAIRS])
def test_references_agree_with_each_other(a, b):
    failures, compared = [], 0
    for key in set(_cases(a)) & set(_cases(b)):
        fam, rot, pars = key
        ca, cb = _cases(a)[key], _cases(b)[key]
        for q in VECTOR_QUANTITIES + SCALAR_QUANTITIES + ("A",):
            if ca.get(q) is None or cb.get(q) is None:
                continue
            if _limited(a, fam, pars, q) or _limited(b, fam, pars, q):
                continue
            x = np.atleast_1d(np.array(ca[q], dtype=float))
            y = np.atleast_1d(np.array(cb[q], dtype=float))
            if q in ("h1", "h2") and "vinecopula" in (a, b):
                x, y = np.clip(x, *VINECOPULA_H_RANGE), np.clip(y, *VINECOPULA_H_RANGE)
            err = np.abs(x - y) / (np.abs(y) if q in ("pdf", "A") else 1.0)
            compared += err.size
            if not err.max() <= REFERENCE_TOL[q]:
                failures.append(f"{fam}/{rot} {pars} {q}: {err.max():.2e}")
    assert compared > 0
    assert not failures, "\n".join(failures)


# --------------------------------------------------------------------------
# mpmath oracle: the textbook CDF of each family, everything else from it
# --------------------------------------------------------------------------

def _mp_student_cdf(n, x):
    """T_n(x) = 1 − ½ I_{n/(n + x²)}(n/2, ½) for x ≥ 0 (DLMF 8.17)."""
    tail = mp.betainc(n / 2, mp.mpf(1) / 2, 0, n / (n + x * x), regularized=True) / 2
    return 1 - tail if x >= 0 else tail


def _mp_base_cdf(family: str, pars: tuple):
    """The textbook CDF of the unrotated family (the README's key)."""
    p = [mp.mpf(x) for x in pars]
    if family == "galambos":          # Galambos (1975)
        def C(u, v):
            w, z = -mp.log(u), -mp.log(v)
            return u * v * mp.exp((w ** -p[0] + z ** -p[0]) ** (-1 / p[0]))
    elif family == "husler_reiss":    # Hüsler & Reiss (1989)
        def C(u, v):
            w, z, lam = -mp.log(u), -mp.log(v), p[0]
            return mp.exp(-w * mp.ncdf(1 / lam + lam / 2 * mp.log(w / z))
                          - z * mp.ncdf(1 / lam + lam / 2 * mp.log(z / w)))
    elif family == "tev":             # Demarta & McNeil (2005)
        def C(u, v):
            w, z = -mp.log(u), -mp.log(v)
            rho, nu = p
            k = mp.sqrt((nu + 1) / (1 - rho ** 2))
            return mp.exp(-(w * _mp_student_cdf(nu + 1, k * ((w / z) ** (1 / nu) - rho))
                            + z * _mp_student_cdf(nu + 1, k * ((z / w) ** (1 / nu) - rho))))
    elif family == "tawn":            # Tawn (1988), asymmetric logistic
        def C(u, v):
            w, z = -mp.log(u), -mp.log(v)
            th, pu, pv = p
            return mp.exp(-((1 - pu) * w + (1 - pv) * z + ((pu * w) ** th + (pv * z) ** th) ** (1 / th)))
    elif family == "plackett":        # Plackett (1965)
        def C(u, v):
            th = p[0]
            S = 1 + (th - 1) * (u + v)
            return (S - mp.sqrt(S * S - 4 * th * (th - 1) * u * v)) / (2 * (th - 1))
    elif family == "amh":             # Nelsen (2006), (4.2.3)
        def C(u, v):
            return u * v / (1 - p[0] * (1 - u) * (1 - v))
    elif family == "fgm":
        def C(u, v):
            return u * v * (1 + p[0] * (1 - u) * (1 - v))
    elif family == "a12":             # Nelsen (2006), (4.2.12)
        def C(u, v):
            return 1 / (1 + ((1 / u - 1) ** p[0] + (1 / v - 1) ** p[0]) ** (1 / p[0]))
    elif family == "a14":             # Nelsen (2006), (4.2.14)
        def C(u, v):
            th = p[0]
            return (1 + ((u ** (-1 / th) - 1) ** th + (v ** (-1 / th) - 1) ** th) ** (1 / th)) ** -th
    else:
        raise KeyError(family)
    return C


def _mp_copula_cdf(family: str, rotation: int, pars: tuple):
    C = _mp_base_cdf(family, pars)
    return {0: C,
            90: lambda u, v: v - C(1 - u, v),
            270: lambda u, v: u - C(u, 1 - v)}[rotation]


def _mp_pickands(family: str, pars: tuple):
    """A(t) in the u-share convention: ℓ(t, 1 − t) = −log C(e^{−t}, e^{t−1})."""
    C = _mp_base_cdf(family, pars)
    return lambda t: -mp.log(C(mp.exp(-t), mp.exp(t - 1)))


def _mp_root(f, start: float):
    x0 = mp.mpf(start)
    root = mp.findroot(f, (x0, x0 * (1 + mp.mpf("1e-9"))), solver="secant")
    assert abs(f(root)) < mp.mpf(10) ** (-(mp.mp.dps - 10))
    return root


def _mp_value(family: str, rotation: int, pars: tuple, q: str, a: float, b: float, start: float):
    """q at (a, b) from the textbook CDF alone: derivatives by mp.diff, inverses by root finding.

    The density is a mixed difference of CDF values of order at most 1,
    which loses about log10(1/c) digits where the density is tiny
    (Hüsler–Reiss λ = 5 at (0.01, 0.99): c = 5.7e-51): the precision is
    raised by that much — ``start``, pmcprg's own value, only sets the
    precision — and the result is recomputed 15 digits higher and must
    agree.
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
    """Kendall's τ, independently of every closed form of the package.

    Extreme-value families: the Genest & MacKay (1986) identity
    τ = ∫₀¹ t(1 − t) A''(t)/A(t) dt, A'' by mp.diff of the textbook A (split
    at the kink of the θ → ∞ limit). Archimedean families:
    τ = 1 + 4 ∫₀¹ φ(t)/φ'(t) dt (Genest & MacKay 1986), φ' by mp.diff.
    Plackett and FGM: Hoeffding's τ = 1 − 4 ∫∫ ∂C/∂u ∂C/∂v, with the
    textbook CDF differentiated. A 90°/270° rotation flips the sign.
    """
    with mp.workdps(20):     # 20 digits: the same errors as 30 to 8 digits, 2-4x faster
        return _mp_tau_at_precision(family, rotation, pars)


def _mp_tau_at_precision(family: str, rotation: int, pars: tuple):
    sign = -1 if rotation in (90, 270) else 1
    p = [mp.mpf(x) for x in pars]
    if family in EV_FAMILIES:
        A = _mp_pickands(family, pars)
        kink = p[2] / (p[1] + p[2]) if family == "tawn" else mp.mpf(1) / 2
        return mp.quad(lambda t: t * (1 - t) * mp.diff(A, t, 2) / A(t), [0, kink, 1])
    phi = {"amh": lambda t: mp.log((1 - p[0] * (1 - t)) / t),
           "a12": lambda t: (1 / t - 1) ** p[0],
           "a14": lambda t: (t ** (-1 / p[0]) - 1) ** p[0]}.get(family)
    if phi is not None:
        return sign * (1 + 4 * mp.quad(lambda t: phi(t) / mp.diff(phi, t), [0, 1]))
    if family == "plackett":
        th = p[0]

        def h(y, x):             # ∂C/∂x at (x, y)
            S = 1 + (th - 1) * (x + y)
            return (1 - (S - 2 * th * y) / mp.sqrt(S * S - 4 * th * (th - 1) * x * y)) / 2
    elif family == "fgm":
        def h(y, x):
            return y * (1 + p[0] * (1 - y) * (1 - 2 * x))
    else:
        raise KeyError(family)
    return 1 - 4 * mp.quad(lambda x, y: h(y, x) * h(x, y), [0, 1], [0, 1])


def _mp_tail_dependence(family: str, rotation: int, pars: tuple):
    """(λ_L, λ_U): 2 − 2A(½) for the EV families (λ_L = 0); otherwise the
    diagonal limits C(u, u)/u and (1 − 2u + C(u, u))/(1 − u), evaluated at
    u = 10⁻⁸⁰⁰ and 1 − 10⁻⁴⁰ (A14's lower ratio converges like u^{1/θ})."""
    if family in EV_FAMILIES:
        return mp.mpf(0), 2 - 2 * _mp_pickands(family, pars)(mp.mpf(1) / 2)
    C = _mp_copula_cdf(family, rotation, pars)
    with mp.workdps(120):
        lo = mp.mpf(10) ** -800
        hi = 1 - mp.mpf(10) ** -40
        return C(lo, lo) / lo, (1 - 2 * hi + C(hi, hi)) / (1 - hi)


def _oracle_tolerance(family: str, rotation: int, q: str) -> float:
    return PMCPRG_LIMITED.get((family, rotation, q), ORACLE_TOL[q])


# t-EV's Student CDF (mp.betainc) makes its 900 oracle values 1.7 s: slow.
ORACLE_VALUE_PARAMS = [pytest.param(f, r, id=f"{f}{r}",
                                    marks=[pytest.mark.slow] if f == "tev" else [])
                       for f, r in FAMILY_ROTATIONS]


@pytest.mark.parametrize("family, rotation", ORACLE_VALUE_PARAMS)
def test_mpmath_oracle_values(family, rotation):
    """pdf, cdf, h1, h2, hinv1, hinv2 of every case at every point, against mpmath (30 digits)."""
    pytest.importorskip("mpmath")
    failures = []
    for fam, rot, pars in CASE_KEYS:
        if (fam, rot) != (family, rotation):
            continue
        for member in _members(fam, rot, pars):
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
def test_mpmath_oracle_tau_tail_pickands(family, rotation):
    """τ, λ_L, λ_U and A(t) of every case against mpmath (τ at 20 digits, the rest 30)."""
    pytest.importorskip("mpmath")
    failures = []
    for fam, rot, pars in CASE_KEYS:
        if (fam, rot) != (family, rotation):
            continue
        for member in _members(fam, rot, pars):
            ours = _ours(member, fam, rot, pars)
            with mp.workdps(30):
                exact = {"tau": _mp_tau(fam, rot, pars)}
                exact["lambda_L"], exact["lambda_U"] = _mp_tail_dependence(fam, rot, pars)
                for q, value in exact.items():
                    err = float(abs(ours[q] - value))
                    if not err <= _oracle_tolerance(fam, rot, q):
                        failures.append(f"{member} {q} {pars}: pmcprg {ours[q]!r}, mpmath {value}")
                if ours["A"] is not None:
                    A = _mp_pickands(fam, pars)
                    err = max(float(abs(x - A(mp.mpf(t))) / A(mp.mpf(t))) for x, t in zip(ours["A"], TGRID))
                    if not err <= ORACLE_TOL["A"]:
                        failures.append(f"{member} A {pars}: {err:.2e}")
    assert not failures, "\n".join(failures)


def _worst(entry: Limited, q: str):
    """(error, key, point index) of the largest |pmcprg − package| in the entry's scope."""
    worst = (-1.0, None, None)
    for key, case in _cases(entry.reference).items():
        fam, rot, pars = key
        if not entry.covers(entry.reference, fam, pars, q):
            continue
        ours = _ours(_members(*key)[0], fam, rot, pars)
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
    ids=[f"{e.reference}-{e.family}-{'all' if e.pars is None else e.pars[0][0]}-{q}"
         for e, q in LIMITED_PARAMS])
def test_reference_limited_entries_are_the_reference(entry, q):
    """At the worst point of each entry, mpmath sides with pmcprg, not with the package."""
    pytest.importorskip("mpmath")
    err, key, k = _worst(entry, q)
    assert key is not None
    fam, rot, pars = key
    # Still needed (else the entry must go), and within what the entry allows.
    assert _tolerance("none", fam, rot, pars, q) < err <= entry.tol, (err, entry)
    ours = _ours(_members(*key)[0], fam, rot, pars)
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
    # the discrepancy (measured: pmcprg ≤ 4.0e-14 relative / 1.8e-12 on τ at
    # these points, the packages 10⁻¹² to O(1)).
    assert ours_err <= _oracle_tolerance(fam, rot, q), (ours_err, entry)
    assert ref_err >= 0.9 * err, (ref_err, err, entry)


@pytest.mark.parametrize("cell", sorted(PMCPRG_LIMITED), ids=lambda c: f"{c[0]}{c[1]}-{c[2]}")
def test_pmcprg_limited_cells_are_pmcprg(cell):
    """Each PMCPRG_LIMITED cell is still needed, and a package that has the
    quantity (copula's AMH inverse) is accurate where pmcprg is not."""
    pytest.importorskip("mpmath")
    family, rotation, q = cell
    worst = (-1.0, None, None)
    for fam, rot, pars in CASE_KEYS:
        if (fam, rot) != (family, rotation):
            continue
        ours = _ours(_members(fam, rot, pars)[0], fam, rot, pars)
        with mp.workdps(30):
            if q == "tau":
                err, k = float(abs(ours["tau"] - _mp_tau(fam, rot, pars))), None
                if err > worst[0]:
                    worst = (err, pars, k)
                continue
            for k, (a, b) in enumerate(POINTS):
                err = float(abs(ours[q][k] - _mp_value(fam, rot, pars, q, a, b, start=ours[q][k])))
                if err > worst[0]:
                    worst = (err, pars, k)
    err, pars, k = worst
    assert ORACLE_TOL[q] < err <= PMCPRG_LIMITED[cell], (cell, err)
    for name in REFERENCES:
        case = _cases(name).get((family, rotation, pars))
        if case is None or case.get(q) is None or _limited(name, family, pars, q):
            continue
        with mp.workdps(30):
            a, b = POINTS[k]
            exact = _mp_value(family, rotation, pars, q, a, b, start=case[q][k])
            # copula's AMH inverse: 7.3e-15 from mpmath (uniroot, tol = 1e-15)
            assert float(abs(case[q][k] - exact)) < 1e-14, (name, cell)


def test_copula_tev_tau_is_the_nu4_curve_of_rho_squared():
    """Why copula's t-EV τ is off: ``tevTauFun`` is a spline in ρ² of the ν = 4
    curve — the ``df`` of the copula and the sign of ρ never reach it."""
    recorded = {pars: case["tau"] for (f, _, pars), case in _cases("rcopula").items() if f == "tev"}
    for (rho, nu), tau in recorded.items():
        # pmcprg's τ(|ρ|, ν = 4), 7e-15 from mpmath; the spline's own error
        # is 2.5e-7 at ρ = 0.8.
        assert tau == pytest.approx(t_ev._tau_of(_tev_s(abs(rho)), 4.0), abs=5e-7)
        if (rho, nu) != (0.5, 4.0):
            assert abs(tau - _ours("TEV", "tev", 0, (rho, nu))["tau"]) > 0.05
