"""Copula evaluation and estimation at the edges — acceptance tests of FR-1..FR-3.

A robustness audit of the copulas found defects that the existing numerical tests could not
see, because their grids stop at |τ| = 0.15 and at u = 0.005: Clayton cancels
catastrophically near independence (RB-1), Joe has no θ for τ < 5·10⁻⁷ (RB-2),
the two-parameter fits of ICE stop at their start value (RB-3), Student's
log-density is floored at −36 nat (RB-4), and objectives are not protected
against 0·(−∞) (RB-7).

The references here do not come from another library — R ``copula``,
VineCopula and vinecopulib clamp u to [10⁻¹², 1 − 10⁻¹²] or cap θ, so they are
not oracles in the tails. They are computed with Python's ``decimal``
directly from the published copula CDFs (Nelsen 2006, Table 4.1; Joe 1997),
the density and h-function being taken as mixed and first finite differences
of C at a step 10⁻²⁰ of the distance to the edge — independent of every
formula in the package. The working precision is not fixed: a mixed
difference of C at (10⁻¹², 1 − 10⁻¹²) cancels about a hundred digits, so the
precision is raised until two successive precisions agree to 12 digits.
Gaussian and Student references use scipy's quantile functions and the
elliptical log-density written with ``log1p``.

Computing the ``decimal`` references takes minutes, so the tests read them
from ``data/copula_limits_references.json``: the float64 values the tests
compare with (shortest round-trip ``repr``), with the precision reached and the
convergence status, keyed by family, θ, δ and (u, v). The ``decimal`` code
stays here and guards the file: ``test_reference_file_sample_recomputes_identically``
recomputes a seeded sample of rows on every run, ``test_reference_file_covers_exactly_the_cases``
checks the file holds exactly the (family, θ, δ) the cases build, and
``PMC_REGEN_COPULA_REFS=1`` (or ``python scripts/regen_copula_references.py
--check``) recomputes every row and fails if the file differs. A case missing
from the file is computed in ``decimal`` at run time, with a warning.
"""
from __future__ import annotations

import functools
import json
import math
import os
import random
import warnings
from decimal import Decimal as D, getcontext, localcontext
from pathlib import Path

import numpy as np
import pytest
from scipy import special, stats

from pmcprg.copulas import CopulaEnum
from pmcprg.exceptions import CopulaError

# --------------------------------------------------------------------------
# High-precision reference CDFs
# --------------------------------------------------------------------------

_ONE = D(1)


def _pw(x: D, a: D) -> D:
    return (a * x.ln()).exp()


def _c_clayton(u, v, th, _):
    return _pw(_pw(u, -th) + _pw(v, -th) - _ONE, -_ONE / th)


def _c_gh(u, v, th, _):
    s = _pw(-u.ln(), th) + _pw(-v.ln(), th)
    return (-_pw(s, _ONE / th)).exp()


def _c_joe(u, v, th, _):
    a, b = _pw(_ONE - u, th), _pw(_ONE - v, th)
    return _ONE - _pw(a + b - a * b, _ONE / th)


def _c_frank(u, v, th, _):
    num = ((-th * u).exp() - _ONE) * ((-th * v).exp() - _ONE)
    return -(_ONE + num / ((-th).exp() - _ONE)).ln() / th


def _c_amh(u, v, th, _):
    return u * v / (_ONE - th * (_ONE - u) * (_ONE - v))


def _c_plackett(u, v, th, _):
    if th == _ONE:
        return u * v
    s = _ONE + (th - _ONE) * (u + v)
    return (s - (s * s - 4 * th * (th - _ONE) * u * v).sqrt()) / (2 * (th - _ONE))


def _c_fgm(u, v, th, _):
    return u * v * (_ONE + th * (_ONE - u) * (_ONE - v))


def _c_a12(u, v, th, _):
    return _ONE / (_ONE + _pw(_pw(_ONE / u - _ONE, th) + _pw(_ONE / v - _ONE, th), _ONE / th))


def _c_a14(u, v, th, _):
    uu = _pw(_pw(u, -_ONE / th) - _ONE, th)
    vv = _pw(_pw(v, -_ONE / th) - _ONE, th)
    return _pw(_ONE + _pw(uu + vv, _ONE / th), -th)


def _c_bb1(u, v, th, de):
    a, b = _pw(_pw(u, -th) - _ONE, de), _pw(_pw(v, -th) - _ONE, de)
    return _pw(_ONE + _pw(a + b, _ONE / de), -_ONE / th)


def _survival(base):
    return lambda u, v, th, de: u + v - _ONE + base(_ONE - u, _ONE - v, th, de)


def _rotated_90(base):
    """C90(u,v) = v − C_base(1−u, v) (FR-8; ``pmcprg.copulas.archimedean.rotated``)."""
    return lambda u, v, th, de: v - base(_ONE - u, v, th, de)


def _rotated_270(base):
    """C270(u,v) = u − C_base(u, 1−v) (FR-8; ``pmcprg.copulas.archimedean.rotated``)."""
    return lambda u, v, th, de: u - base(u, _ONE - v, th, de)


def _c_galambos(u, v, th, _):
    w, z = -u.ln(), -v.ln()
    s = _pw(_pw(w, -th) + _pw(z, -th), -_ONE / th)
    return u * v * s.exp()


def _ndtr_decimal(x: D) -> D:
    """Standard normal CDF Φ(x) as a ``Decimal``, at the *current* decimal
    context's precision.

    ``decimal`` has no erf/normal-CDF of its own, unlike every other
    family's reference CDF above (elementary functions only); ``mpmath``
    is this reference's ground truth for Φ, exactly as the FR-9
    Hüsler-Reiss module's own docstring uses it — imported lazily so a
    ``mpmath``-less install (the ``min-versions`` CI job, which does not
    install the ``dev`` extra) only fails if this is actually called, which
    it is not on the fast suite once the reference file below covers every
    case (``_decimal_records`` reads the file first).
    """
    import mpmath
    prec = getcontext().prec
    mpmath.mp.dps = prec + 15   # guard digits beyond the decimal context's own
    x_mp = mpmath.mpf(str(x))
    phi = (1 + mpmath.erf(x_mp / mpmath.sqrt(2))) / 2
    return D(mpmath.nstr(phi, prec + 10, strip_zeros=False))


def _c_husler_reiss(u, v, lam, _):
    """Hüsler-Reiss CDF, C(u,v) = exp(-(w·Φ(g) + z·Φ(h))) — see
    ``pmcprg.copulas.extreme_value.husler_reiss`` module docstring for the
    derivation and its own independent ``mpmath`` cross-checks; this is an
    independent re-implementation (not a shared helper import) so the
    reference does not silently inherit a bug from the family's own code."""
    w, z = -u.ln(), -v.ln()
    r = w.ln() - z.ln()
    two = D(2)
    g = _ONE / lam + lam / two * r
    h = _ONE / lam - lam / two * r
    neg_log_c = w * _ndtr_decimal(g) + z * _ndtr_decimal(h)
    return (-neg_log_c).exp()


def _tawn(u_fixed):
    """Tawn type 1 (``u_fixed``: ψ_u = 1, ψ_v = ψ) or type 2 (ψ_u = ψ, ψ_v = 1)
    CDF, C = exp(−ℓ), ℓ = (1 − ψ_u)w + (1 − ψ_v)z + ((ψ_u w)^θ + (ψ_v z)^θ)^{1/θ}
    (FR-9, ``pmcprg.copulas.extreme_value.tawn``) — written from the model's
    definition, not from the module's log-space kernel. The ``δ`` slot of the
    reference machinery carries ψ for these two families."""
    def c(u, v, th, psi):
        pu, pv = (_ONE, psi) if u_fixed else (psi, _ONE)
        w, z = -u.ln(), -v.ln()
        n = _pw(_pw(pu * w, th) + _pw(pv * z, th), _ONE / th)
        return (-((_ONE - pu) * w + (_ONE - pv) * z + n)).exp()
    return c


_REF_C = {
    "Clayton": _c_clayton, "GH": _c_gh, "Joe": _c_joe, "Frank": _c_frank,
    "AMH": _c_amh, "Plackett": _c_plackett, "FGM": _c_fgm, "A12": _c_a12,
    "A14": _c_a14, "BB1": _c_bb1, "SClayton": _survival(_c_clayton),
    "SGH": _survival(_c_gh), "SJoe": _survival(_c_joe),
    "Galambos": _c_galambos, "HuslerReiss": _c_husler_reiss,
    "Clayton90": _rotated_90(_c_clayton), "Clayton270": _rotated_270(_c_clayton),
    "GH90": _rotated_90(_c_gh), "GH270": _rotated_270(_c_gh),
    "Joe90": _rotated_90(_c_joe), "Joe270": _rotated_270(_c_joe),
    "BB190": _rotated_90(_c_bb1), "BB1270": _rotated_270(_c_bb1),
    "Tawn1": _tawn(True), "Tawn2": _tawn(False),
}


def _ref_at_prec(C, u, v, th, de, prec):
    with localcontext() as ctx:
        ctx.prec = prec
        U, V, TH = D(u), D(v), D(th)
        DE = D(de) if de is not None else None
        hu = min(U, _ONE - U) * D("1e-20")
        hv = min(V, _ONE - V) * D("1e-20")
        c0 = C(U, V, TH, DE)
        mixed = (C(U + hu, V + hv, TH, DE) - C(U + hu, V - hv, TH, DE)
                 - C(U - hu, V + hv, TH, DE) + C(U - hu, V - hv, TH, DE)) / (4 * hu * hv)
        h = (C(U + hu, V, TH, DE) - C(U - hu, V, TH, DE)) / (2 * hu)
        return c0, mixed, h


_PRECISIONS = (120, 240, 480, 960, 1920)


def _ref_record(short, u, v, th, de):
    """C, log c and h(v|u) = ∂C/∂u at (u, v) as floats, the last precision
    used and whether two successive precisions agreed to 12 digits.

    The precision needed is the digits cancelled by the difference quotient,
    ≈ log10(C / (h_u h_v c)), plus the digits wanted: it is raised until two
    successive precisions agree, rather than guessed.
    """
    C = _REF_C[short]
    prev = None
    converged = False
    for prec in _PRECISIONS:
        c0, mixed, h = _ref_at_prec(C, u, v, th, de, prec)
        cur = (c0, mixed, h)
        # Relative agreement only. An absolute floor would accept a difference
        # quotient that is still pure cancellation noise: at (0.5, 1e-12) for
        # survival Joe at τ = 0.95 the mixed difference is exactly 0 at 240
        # digits and e^-986 at 480 — both wrong (log c = −1011.89 at 960 and
        # 1920 digits). An exact 0 is accepted only once it survives 960 digits
        # (a genuine zero density, e.g. the cubic-section corner at θ = 1/4).
        if prev is not None and all(
            (a == 0 and b == 0 and prec >= 960) or (b != 0 and abs(a - b) <= abs(b) * D("1e-12"))
            for a, b in zip(cur, prev)
        ):
            converged = True
            break
        prev = cur
    logc = float(mixed.ln()) if mixed > 0 else -math.inf
    return float(c0), logc, float(h), prec, converged


def _ref_logc_elliptical(short, u, v, rho, df):
    def z(p):
        if short == "Gauss":
            return special.ndtri(p) if p <= 0.5 else -special.ndtri(1.0 - p)
        return stats.t.ppf(p, df) if p <= 0.5 else -stats.t.ppf(1.0 - p, df)
    x, y = z(u), z(v)
    r2 = 1.0 - rho * rho
    if short == "Gauss":
        return -(rho * rho * (x * x + y * y) - 2 * rho * x * y) / (2 * r2) - 0.5 * math.log1p(-rho * rho)
    q = (x * x - 2 * rho * x * y + y * y) / (df * r2)
    log2 = (special.gammaln((df + 2) / 2) - special.gammaln(df / 2) - math.log(df * math.pi)
            - 0.5 * math.log(r2) - (df + 2) / 2 * math.log1p(q))
    def log1(t):
        return (special.gammaln((df + 1) / 2) - special.gammaln(df / 2) - 0.5 * math.log(df * math.pi)
                - (df + 1) / 2 * math.log1p(t * t / df))
    return log2 - log1(x) - log1(y)


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------

_ENTRY = {e.value.SHORT_NAME: e for e in CopulaEnum if e.value.AVAILABLE}

# τ values per family: moderate, strong, and the independence limit where the
# family contains independence.
_TAIL_TAUS = {
    "Clayton": (0.3, 0.7, 0.95), "SClayton": (0.3, 0.7, 0.95),
    "GH": (0.3, 0.7, 0.95), "SGH": (0.3, 0.7, 0.95),
    "Joe": (0.3, 0.7, 0.95), "SJoe": (0.3, 0.7, 0.95),
    "Frank": (-0.7, 0.3, 0.9), "Plackett": (-0.7, 0.3, 0.9),
    "AMH": (-0.15, 0.2, 0.3), "FGM": (-0.2, 0.2),
    "A12": (0.4, 0.7, 0.9), "A14": (0.4, 0.7, 0.9), "BB1": (0.4, 0.7, 0.9),
    "Gauss": (-0.7, 0.3, 0.9), "Student": (-0.7, 0.3, 0.95),
    "Galambos": (0.3, 0.7, 0.95), "HuslerReiss": (0.3, 0.7, 0.95),
    "Clayton90": (-0.3, -0.7, -0.95), "Clayton270": (-0.3, -0.7, -0.95),
    "GH90": (-0.3, -0.7, -0.95), "GH270": (-0.3, -0.7, -0.95),
    "Joe90": (-0.3, -0.7, -0.95), "Joe270": (-0.3, -0.7, -0.95),
    "BB190": (-0.4, -0.7, -0.9), "BB1270": (-0.4, -0.7, -0.9),
    "Tawn1": (0.3, 0.7, 0.95), "Tawn2": (0.3, 0.7, 0.95),
}
_INDEP_TAUS = {
    "Clayton": (1e-12, 1e-8, 1e-4), "SClayton": (1e-12, 1e-8, 1e-4),
    "GH": (1e-12, 1e-8, 1e-4), "SGH": (1e-12, 1e-8, 1e-4),
    "Joe": (1e-12, 1e-8, 1e-4), "SJoe": (1e-12, 1e-8, 1e-4),
    "Frank": (-1e-8, 1e-12, 1e-4), "Plackett": (-1e-8, 1e-12, 1e-4),
    "AMH": (-1e-8, 1e-12, 1e-4), "FGM": (-1e-8, 1e-12, 1e-4),
    "Gauss": (-1e-8, 1e-12, 1e-4),
    "Galambos": (1e-12, 1e-8, 1e-4), "HuslerReiss": (1e-12, 1e-8, 1e-4),
    "Clayton90": (-1e-4, -1e-8, -1e-12), "Clayton270": (-1e-4, -1e-8, -1e-12),
    "GH90": (-1e-4, -1e-8, -1e-12), "GH270": (-1e-4, -1e-8, -1e-12),
    "Joe90": (-1e-4, -1e-8, -1e-12), "Joe270": (-1e-4, -1e-8, -1e-12),
    "Tawn1": (1e-12, 1e-8, 1e-4), "Tawn2": (1e-12, 1e-8, 1e-4),
}

# ψ of the Tawn cases (FR-9), per τ: it must exceed τ (the reachable-τ cap).
# Strong asymmetry close to the cap (τ/ψ = 0.86 and 0.98, large θ), a weak
# one (0.7, 0.9), and small ψ at the independence end.
_TAWN_PSI = {0.3: 0.35, 0.7: 0.9, 0.95: 0.97, 1e-12: 1e-3, 1e-8: 0.05, 1e-4: 0.5}

_G = (1e-12, 1e-6, 0.3, 0.5, 1 - 1e-6, 1 - 1e-12)
_UV = np.array([(u, v) for u in _G for v in _G])


def _build(short, tau, df=4.0, delta=1.5):
    kw = {"tau_k": tau}
    names = _ENTRY[short].value.PARAMETERS_SET_NAME
    if "df" in names:
        kw["df"] = df
    if "delta" in names:
        kw["delta"] = delta
    if "psi" in names:
        kw["psi"] = _TAWN_PSI.get(tau, 1.0)
    return _ENTRY[short].klass(**kw)


def _second_param(cop):
    """The file's ``delta`` slot: BB1's δ, Tawn's ψ, else ``None``."""
    return getattr(cop, "delta", getattr(cop, "psi", None))


_CASES = [pytest.param(s, t, id=f"{s}-tau{t:+.2g}")
          for s, ts in _TAIL_TAUS.items() for t in ts]
_CASES += [pytest.param(s, t, id=f"{s}-indep{t:+.0e}")
           for s, ts in _INDEP_TAUS.items() for t in ts]
_CASES += [pytest.param("Student", 0.95, id="Student-tau+0.95-df30")]


# --------------------------------------------------------------------------
# Reference file — the decimal references above, computed once
# --------------------------------------------------------------------------
# Only the decimal references are stored: the elliptical ones cost a few
# scipy calls and depend on scipy, so they stay computed at run time.
#
# Encoding: each float is the shortest string that round-trips to the IEEE-754
# double the tests compare with (``repr``), so reading the file gives exactly
# the references the tests used to compute (a decimal string of 12 digits would
# not, and float.hex is exact too but unreadable). The recomputation is itself
# deterministic — decimal arithmetic, ln, exp and sqrt are correctly rounded,
# and so is float(Decimal) — hence the checks below compare exactly. One entry
# per (family, θ, δ), one row per (u, v) of the grid; entries sorted by family,
# θ, δ and rows by (u, v), so regenerating unchanged references rewrites the
# same bytes.

_REF_FILE = Path(__file__).resolve().parent / "data" / "copula_limits_references.json"
_REF_FORMAT = 1
_REF_COLUMNS = ["u", "v", "C", "log_c", "h", "precision", "converged"]
_REGEN_ENV = "PMC_REGEN_COPULA_REFS"
_REGEN_HINT = "regenerate it with `python scripts/regen_copula_references.py`"
_ELLIPTICAL = ("Gauss", "Student")
# Drift guard: rows recomputed on every run, drawn (seeded) within strata —
# one 240-digit row per family (≈ 93 % of the rows converge at 240 digits),
# and a few rows at each higher precision reached, so the costly adaptive
# steps are exercised too. Cost of one row: ≈ 7 ms at 240 digits, 0.07 s at
# 480, 0.4 s at 960, 3 s at 1920 — 19 rows, about 4 s in all.
_SAMPLE_SEED = 20260915
_SAMPLE_PER_PRECISION = {240: 1, 480: 3, 960: 2, 1920: 1}  # lowest precision in the file: per family


def _num(x) -> str:
    """A float as the file stores it: its shortest round-trip repr."""
    return repr(float(x))


def _param_key(short, th, de):
    return (short, _num(th), None if de is None else _num(de))


def _param_sort_key(key):
    short, th, de = key
    return (short, float(th), -math.inf if de is None else float(de))


def _grid_args():
    return [(_num(u), _num(v)) for u, v in _UV]


def _required_param_keys():
    """(family, θ, δ) of every non-elliptical case, built as the tests build it."""
    keys = set()
    for case in _CASES:
        short, tau = case.values
        if short not in _ELLIPTICAL:
            cop = _build(short, tau)
            keys.add(_param_key(short, cop.theta, _second_param(cop)))
    return keys


def _record_task(task):
    """One file row [u, v, C, log c, h, precision, converged], computed in decimal."""
    short, th, de, u, v = task
    c0, logc, h, prec, converged = _ref_record(
        short, float(u), float(v), float(th), None if de is None else float(de))
    return [u, v, _num(c0), _num(logc), _num(h), prec, converged]


def _compute_reference_table(keys=None, mapper=map):
    """{(family, θ, δ): rows} recomputed in decimal, for ``keys`` (default: every case).

    ``mapper`` is ``map`` or an order-preserving parallel map (the script's).
    """
    keys = sorted(_required_param_keys() if keys is None else keys, key=_param_sort_key)
    grid = _grid_args()
    rows = iter(mapper(_record_task, [(*key, u, v) for key in keys for u, v in grid]))
    return {key: [next(rows) for _ in grid] for key in keys}


def _dump_reference_table(table) -> str:
    """The canonical text of the file: sorted, one row per line."""
    keys = sorted(table, key=_param_sort_key)
    lines = [
        "{",
        f'  "format": {_REF_FORMAT},',
        '  "about": "High-precision decimal references of pmcprg/tests/test_copula_limits.py; '
        'generated by scripts/regen_copula_references.py, do not edit by hand. Floats are '
        'shortest round-trip repr strings of IEEE-754 doubles; precision is the last decimal '
        'precision used, converged whether two successive precisions agreed to 12 digits.",',
        f'  "precisions": {json.dumps(list(_PRECISIONS))},',
        f'  "columns": {json.dumps(_REF_COLUMNS)},',
        '  "entries": [',
    ]
    for i, key in enumerate(keys):
        short, th, de = key
        lines.append(f'    {{"family": {json.dumps(short)}, "theta": {json.dumps(th)}, '
                     f'"delta": {json.dumps(de)}, "points": [')
        rows = sorted(table[key], key=lambda r: (float(r[0]), float(r[1])))
        lines += [f"      {json.dumps(row)}{',' if j < len(rows) - 1 else ''}" for j, row in enumerate(rows)]
        lines.append(f"    ]}}{',' if i < len(keys) - 1 else ''}")
    lines += ["  ]", "}"]
    return "\n".join(lines) + "\n"


def _parse_reference_table(text):
    doc = json.loads(text)
    if doc.get("format") != _REF_FORMAT or doc.get("columns") != _REF_COLUMNS:
        raise ValueError(f"unsupported reference file format {doc.get('format')!r}")
    table = {}
    for entry in doc["entries"]:
        key = (entry["family"], entry["theta"], entry["delta"])
        if key in table:
            raise ValueError(f"duplicate reference entry {key}")
        table[key] = [list(row) for row in entry["points"]]
    return table


def _table_differences(old, new):
    """Human-readable differences between two reference tables."""
    out = [f"missing {key}" for key in sorted(new.keys() - old.keys(), key=_param_sort_key)]
    out += [f"extra {key}" for key in sorted(old.keys() - new.keys(), key=_param_sort_key)]
    for key in sorted(old.keys() & new.keys(), key=_param_sort_key):
        old_rows = {(r[0], r[1]): r for r in old[key]}
        for row in new[key]:
            if old_rows.get((row[0], row[1])) != row:
                out.append(f"{key} at (u, v) = ({row[0]}, {row[1]}): file {old_rows.get((row[0], row[1]))}, recomputed {row}")
        if len(old_rows) != len(new[key]):
            out.append(f"{key}: {len(old_rows)} rows in the file, {len(new[key])} recomputed")
    return out


@functools.cache
def _file_table():
    """The reference file as {(family, θ, δ): rows}; empty if the file is absent."""
    try:
        return _parse_reference_table(_REF_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


@functools.cache
def _file_index():
    return {key: {(r[0], r[1]): r for r in rows} for key, rows in _file_table().items()}


# θ of AMH and Plackett (and δ where derived) come out of a numerical inversion
# of τ whose last bits depend on the platform's libm and on the NumPy/SciPy
# version (Linux CI vs macOS, lowest declared versions), so a file entry is
# matched within this relative tolerance, not by its exact repr. The references
# at θ and at θ(1 ± 1e-12) agree far below the tolerances of the tests using them.
_PARAM_RTOL = 1e-12


def _close(a, b) -> bool:
    if a is None or b is None:
        return a is b
    a, b = float(a), float(b)
    return abs(a - b) <= _PARAM_RTOL * max(abs(a), abs(b))


def _match_file_key(key, candidates):
    """The key of ``candidates`` equal to ``key``, else the one within ``_PARAM_RTOL``; None if none."""
    if key in candidates:
        return key
    short, th, de = key
    close = [k for k in candidates if k[0] == short and _close(k[1], th) and _close(k[2], de)]
    return min(close, key=lambda k: abs(float(k[1]) - float(th))) if close else None


def _decimal_records(short, th, de):
    """(C, log c, h, precision, converged) at each point of ``_UV``: read from the
    reference file, or computed in decimal (with a warning) if the case is not in it."""
    index = _file_index()
    match = _match_file_key(_param_key(short, th, de), index)
    rows = None if match is None else index[match]
    if rows is None:
        warnings.warn(f"{_param_key(short, th, de)} is not in {_REF_FILE.name}: computing its "
                      f"references in decimal (slow); {_REGEN_HINT}", stacklevel=2)
        return [_ref_record(short, u, v, th, de) for u, v in _UV]
    out = []
    for u, v in _UV:
        _, _, c0, logc, h, prec, converged = rows[(_num(u), _num(v))]
        out.append((float(c0), float(logc), float(h), prec, converged))
    return out


_REF_CACHE: dict = {}


def _reference(short, tau, cop, df):
    de = _second_param(cop)
    key = (short, float(cop.theta), de, df)
    if key not in _REF_CACHE:
        if short in _ELLIPTICAL:
            logc = np.array([_ref_logc_elliptical(short, u, v, cop.theta, df) for u, v in _UV])
            _REF_CACHE[key] = (None, logc, None)
        else:
            out = []
            for (u, v), (c0, logc, h, _, converged) in zip(_UV, _decimal_records(short, cop.theta, de)):
                if not converged:
                    raise AssertionError(f"reference did not converge at {(short, u, v, cop.theta)}")
                out.append((c0, logc, h))
            _REF_CACHE[key] = tuple(np.array(col) for col in zip(*out))
    return _REF_CACHE[key]


# --------------------------------------------------------------------------
# The reference file is the decimal code's output
# --------------------------------------------------------------------------

def test_reference_file_covers_exactly_the_cases():
    """The file holds one entry per (family, θ, δ) the cases build — none
    missing, none stale — each on the whole grid, in canonical form."""
    table = _file_table()
    assert table, f"{_REF_FILE} is missing: {_REGEN_HINT}"
    need, have = _required_param_keys(), set(table)
    matched = {key: _match_file_key(key, have) for key in need}
    missing = [key for key, m in matched.items() if m is None]
    stale = have - set(matched.values())
    assert not missing and not stale, (
        f"{_REF_FILE.name} does not match the cases ({_REGEN_HINT}): "
        f"missing {sorted(missing, key=_param_sort_key)[:3]}, "
        f"stale {sorted(stale, key=_param_sort_key)[:3]}")
    grid = _grid_args()
    for key, rows in table.items():
        assert [(r[0], r[1]) for r in rows] == grid, f"{key}: the rows are not the (u, v) grid"
    assert _dump_reference_table(table) == _REF_FILE.read_text(encoding="utf-8"), (
        f"{_REF_FILE.name} is not in canonical form: {_REGEN_HINT}")


def test_reference_keys_match_across_platforms():
    """θ values built on GitHub's Linux runners (lowest versions and 3.11–3.14)
    that differ from the file in their last bits still find their entry; a θ
    that differs beyond 1e-12 does not."""
    have = set(_file_table())
    for short, linux_theta, file_theta in [
        ("AMH", "-0.7972341203647484", "-0.7972341203647481"),
        ("Plackett", "0.022675397683867716", "0.022675397683867646"),
        ("Plackett", "3.9949528805878463", "3.994952880587858"),
        ("Plackett", "532.1684249319636", "532.1684249319674"),
    ]:
        assert _match_file_key((short, linux_theta, None), have) == (short, file_theta, None)
        assert _match_file_key((short, _num(float(file_theta) * (1 + 1e-9)), None), have) is None


@pytest.mark.slow
def test_reference_file_sample_recomputes_identically():
    """Drift guard: a seeded random sample of file rows — one per family at the
    lowest precision, a few per higher precision reached and convergence
    status — recomputed in decimal, is identical: values, precision reached and
    convergence status. A drift confined to a few points of one family is left
    to the full regeneration."""
    table = _file_table()
    assert table, f"{_REF_FILE} is missing: {_REGEN_HINT}"
    lowest = min(row[5] for rows in table.values() for row in rows)
    strata: dict = {}
    for key in sorted(table, key=_param_sort_key):
        for row in table[key]:
            prec, converged = row[5], row[6]
            strata.setdefault((prec, converged, key[0] if prec == lowest else ""), []).append((key, row))
    rng = random.Random(_SAMPLE_SEED)
    sample = [item for stratum in sorted(strata)
              for item in rng.sample(strata[stratum], min(len(strata[stratum]), _SAMPLE_PER_PRECISION.get(stratum[0], 1)))]
    bad = []
    for key, row in sample:
        fresh = _record_task((*key, row[0], row[1]))
        if fresh != row:
            bad.append(f"{key} at (u, v) = ({row[0]}, {row[1]}): file {row[2:]}, recomputed {fresh[2:]}")
    assert not bad, f"{len(bad)}/{len(sample)} sampled references changed ({_REGEN_HINT} if intended):\n" + "\n".join(bad)


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get(_REGEN_ENV) != "1",
                    reason=f"recomputes every reference in decimal (minutes); set {_REGEN_ENV}=1")
def test_reference_file_full_regeneration():
    """Opt-in: every reference recomputed in decimal equals the file, byte for byte."""
    fresh = _compute_reference_table()
    diff = _table_differences(_file_table(), fresh)
    assert not diff, f"{len(diff)} differences ({_REGEN_HINT} if intended):\n" + "\n".join(diff[:20])
    assert _dump_reference_table(fresh) == _REF_FILE.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# FR-1 — evaluation against the high-precision references
# --------------------------------------------------------------------------

@pytest.mark.parametrize("short,tau", _CASES)
def test_logpdf_array_matches_high_precision_reference(short, tau, request):
    df = 30.0 if request.node.callspec.id.endswith("df30") else 4.0
    cop = _build(short, tau, df=df)
    _, ref, _ = _reference(short, tau, cop, df)
    got = cop.logpdf_array(_UV)
    finite = np.isfinite(ref)
    assert np.all(np.isfinite(got[finite])), f"non-finite log c where the reference is finite: {_UV[finite & ~np.isfinite(got)][:4]}"
    err = np.abs(got[finite] - ref[finite])
    tol = 1e-7 * np.maximum(1.0, np.abs(ref[finite]))
    bad = err > tol
    assert not bad.any(), (f"{bad.sum()}/{bad.size} points off; worst at "
                           f"{_UV[finite][bad][np.argmax(err[bad])]}: got {got[finite][bad][np.argmax(err[bad])]:.6g}, "
                           f"ref {ref[finite][bad][np.argmax(err[bad])]:.6g}")


@pytest.mark.parametrize("short,tau", _CASES)
def test_pdf_array_is_exp_of_logpdf_array(short, tau, request):
    """No floor on the linear-scale density: pdf_array == exp(logpdf_array)."""
    df = 30.0 if request.node.callspec.id.endswith("df30") else 4.0
    cop = _build(short, tau, df=df)
    logc = cop.logpdf_array(_UV)
    ok = np.isfinite(logc) & (logc > -700.0) & (logc < 700.0)
    np.testing.assert_allclose(cop.pdf_array(_UV)[ok], np.exp(logc[ok]), rtol=1e-12)


@pytest.mark.parametrize("short,tau", _CASES)
def test_scalar_pdf_matches_high_precision_reference(short, tau, request):
    df = 30.0 if request.node.callspec.id.endswith("df30") else 4.0
    cop = _build(short, tau, df=df)
    _, ref, _ = _reference(short, tau, cop, df)
    got = np.array([cop.pdf([u, v]) for u, v in _UV])
    expect = np.exp(np.clip(ref, -745.0, 709.0))
    representable = (ref > -700.0) & (ref < 700.0)
    rel = np.abs(got[representable] - expect[representable]) / expect[representable]
    bad = rel > 1e-6
    assert not bad.any(), (f"{bad.sum()}/{bad.size} points off; worst at "
                           f"{_UV[representable][bad][np.argmax(rel[bad])]}: got {got[representable][bad][np.argmax(rel[bad])]:.6g}, "
                           f"expected {expect[representable][bad][np.argmax(rel[bad])]:.6g}")


@pytest.mark.parametrize("short,tau", [c for c in _CASES if c.values[0] not in ("Gauss", "Student")])
def test_cdf_array_matches_high_precision_reference(short, tau):
    cop = _build(short, tau)
    ref, _, _ = _reference(short, tau, cop, None)
    got = cop.cdf_array(_UV)
    err = np.abs(got - ref)
    bad = err > 1e-13 + 1e-9 * ref
    assert not bad.any(), (f"{bad.sum()}/{bad.size} points off; worst at {_UV[bad][np.argmax(err[bad])]}: "
                           f"got {got[bad][np.argmax(err[bad])]:.17g}, ref {ref[bad][np.argmax(err[bad])]:.17g}")


@pytest.mark.parametrize("short,tau", [c for c in _CASES if c.values[0] not in ("Gauss", "Student")])
def test_h_function_matches_high_precision_reference(short, tau):
    cop = _build(short, tau)
    _, _, ref = _reference(short, tau, cop, None)
    got = np.array([cop.conditional_cdf(v, u) for u, v in _UV])
    err = np.abs(got - ref)
    bad = err > 1e-10 + 1e-8 * ref
    assert not bad.any(), (f"{bad.sum()}/{bad.size} points off; worst at (u, v) = {_UV[bad][np.argmax(err[bad])]}: "
                           f"got {got[bad][np.argmax(err[bad])]:.12g}, ref {ref[bad][np.argmax(err[bad])]:.12g}")


# --------------------------------------------------------------------------
# FR-2 / FR-3 — parameter maps, bounds, fits
# --------------------------------------------------------------------------

@pytest.mark.parametrize("short", sorted(_ENTRY))
def test_registry_bounds_are_constructible_or_refused_explicitly(short):
    """RB-6: at a registered bound a family either works or raises CopulaError —
    never ZeroDivisionError, LinAlgError or a root-finder ValueError."""
    lo, hi = _ENTRY[short].value.TAU_MIN_MAX
    for tau in {lo, hi}:
        try:
            cop = _build(short, tau)
        except CopulaError:
            continue
        vals = cop.logpdf_array(np.array([[0.3, 0.6], [0.5, 0.5]]))
        assert not np.any(np.isnan(vals)), f"{short} at τ={tau}: NaN log-density"


@pytest.mark.parametrize("short", ["Joe", "SJoe"])
@pytest.mark.parametrize("tau", [1e-12, 1e-9, 1e-7, 1e-5])
def test_joe_has_a_parameter_for_every_small_tau(short, tau):
    """RB-2: τ(θ = 1) must be 0, so every τ > 0 has an antecedent."""
    from pmcprg.copulas.archimedean.joe import _joe_tau_from_theta
    assert abs(_joe_tau_from_theta(1.0)) < 1e-13
    cop = _build(short, tau)
    assert abs(_joe_tau_from_theta(cop.theta) - tau) < 1e-10


@pytest.mark.parametrize("short", ["Clayton", "SClayton", "GH", "SGH", "Joe", "SJoe"])
def test_tau_fit_on_negatively_concordant_independent_data_has_null_likelihood(short):
    """RB-1/RB-6: τ̂ ≤ 0 clips to the lower bound, i.e. independence — the
    log-likelihood there is 0, not the +2.86 nat the cancellation produced."""
    seed = 0
    while True:
        uv = np.random.default_rng(seed).random((300, 2))
        if stats.kendalltau(uv[:, 0], uv[:, 1]).statistic < 0:
            break
        seed += 1
    r = _ENTRY[short].klass.fit(uv, method="tau")
    assert abs(r.log_likelihood) < 1e-6, f"LL = {r.log_likelihood}"


@pytest.mark.parametrize("short,tau,fun", [
    ("Frank", 0.999, "kendall_tau_frank"), ("Frank", -0.999, "kendall_tau_frank"),
    ("Plackett", 0.999, "_plackett_tau_from_theta"),
])
def test_stored_tau_is_the_realised_tau(short, tau, fun):
    """RB-10: a clamped copula must not advertise the τ it was asked for."""
    import importlib
    mod = importlib.import_module(_ENTRY[short].klass.__module__)
    cop = _build(short, tau)
    realised = getattr(mod, fun)(cop.theta)
    assert abs(cop.params["tau_k"] - realised) < 1e-6


def test_ice_student_fit_leaves_the_start_value():
    """RB-3: ν = 2.5 must not come back as the start value 4."""
    from pmcprg.pmc.ice import _fit_copula_params, _resolve_candidate
    from pmcprg.copulas.elliptical.student import CopulaStudent
    uv = CopulaStudent(tau_k=0.95, df=2.5).sample(n=1500, seed=11)
    entry, cls = _resolve_candidate("Student")
    p = _fit_copula_params(cls, entry, uv[:, 0], uv[:, 1], np.ones(len(uv)))
    assert 2.0 < p["df"] < 3.4, p


def test_ice_bb1_fit_leaves_the_start_value():
    """RB-3: δ = 3 must not come back as the start value 1.5.

    τ = 0.8 with δ = 3 gives θ = 2/(δ(1 − τ)) − 2 = 4/3 > 0. (The first version
    of this test used τ = 0.6, δ = 3, i.e. θ = −1/3: not a BB1 copula at all.)
    """
    from pmcprg.pmc.ice import _fit_copula_params, _resolve_candidate
    from pmcprg.copulas.archimedean.bb1 import CopulaBB1
    uv = CopulaBB1(tau_k=0.8, delta=3.0).sample(n=1500, seed=11)
    entry, cls = _resolve_candidate("BB1")
    p = _fit_copula_params(cls, entry, uv[:, 0], uv[:, 1], np.ones(len(uv)))
    assert 2.3 < p["delta"] < 3.8, p


def test_weighted_log_likelihood_ignores_non_finite_points_of_zero_weight():
    """RB-7: 0 · (−∞) must not turn the objective into NaN."""
    from pmcprg.pmc.ice import _weighted_log_likelihood

    class _Fake:
        def __init__(self, **_):
            pass

        def logpdf_array(self, uv):
            return np.array([-np.inf, np.nan, -0.5])

    val = _weighted_log_likelihood(_Fake, {"tau_k": 0.5}, np.zeros(3), np.zeros(3), np.array([0.0, 0.0, 2.0]))
    assert val == pytest.approx(-1.0)
