if __name__ == "__main__":
    import sys
    import pathlib

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import logging
import numbers
import os
import matplotlib.pyplot as plt
import numpy as np

from enum import Enum, unique
from dataclasses import dataclass, field

from pmcprg.numerics import EPS, ONE_MINUS_EPS, EPS_MINUS_ONE, minmaxEPS
from pmcprg.plot_style import DEFAULT_FONT_SIZE as FONT_SIZE
from pmcprg.plot_style import with_package_style
from pmcprg.exceptions import CopulaParameterError, CopulaNotAvailableError

# AMH τ_K range constants (computed at import time, θ ∈ [−1, 1))
_AMH_TAU_MIN = float(1.0 - 2.0 * (4.0 * np.log(2.0) - 1.0) / 3.0)  # τ(θ=−1) ≈ −0.1817
_AMH_TAU_MAX = float(1.0 / 3.0 - EPS)                                  # limit τ → 1/3

logger = logging.getLogger(__name__)

# Module-level cache for CopulaEnum.klass (lazy import).
_COPULA_KLASS_CACHE: dict = {}

# Canonical bounds + init for the *non*-τ parameters of multi-parameter copula
# families, keyed by parameter name as ``(lower, upper, init)``. SINGLE SOURCE
# OF TRUTH: standalone fitting (``CopulaVirt.fit`` below) reads it directly, and
# ICE-driven fitting (``pmcprg.pmc.ice.EXTRA_PARAM_BOUNDS``) derives its
# class-keyed view from this dict + the registry — so the two cannot drift
# (audit A-2). Currently: BB1 ``delta``, Student ``df``, Tawn types 1/2
# ``psi`` (FR-9), t-EV ``nu`` (FR-9), BB6 ``delta6`` (FR-9), BB7 ``theta7``
# (FR-9 — θ and not δ, see that module's "Parametrisation"), BB8 ``delta8``
# (FR-9),
# three-parameter Tawn ``psi_u``/``psi_v`` (FR-9 — the first family with
# *two* extra parameters; every consumer of this dict iterates it, so
# nothing here caps their number).
#
# Every bound is itself admissible (audit RB-4/FR-3): the Student copula
# requires ν > 2 and refuses ν = 2.0, where the former lower bound sat — the
# fit then saw the failure penalty (log-likelihood −1e12) at the bound against
# +3.6 nat at 2 + 1e-8. 2.001 rather than 2 + 1e-8 because the GUI edits df in
# a spin box with three decimals, which would round 2 + 1e-8 back to 2.000.
EXTRA_PARAM_BOUNDS_BY_PARAM: dict[str, tuple[float, float, float]] = {
    "delta": (1.0,    10.0,  1.5),
    "df":    (2.001, 100.0,  4.0),
    # Tawn types 1/2 (FR-9): asymmetry ψ ∈ (0, 1], jointly constrained with τ
    # (τ < ψ — ``pmcprg.copulas.extreme_value.tawn``). The init 1.0 is the
    # Gumbel member, admissible at every τ < 1; the lower bound is admissible
    # for τ < 0.01 only, as BB1's δ = 10 is for τ > 0.9 only.
    "psi":   (0.01,    1.0,  1.0),
    # t-EV (FR-9): the parent Student-t's ν, under its own name — not "df",
    # whose bounds are Student's (ν > 2) and whose name ``_two_parameter_spec``
    # dispatches on (``pmcprg.copulas.extreme_value.t_ev``, "Parametrisation").
    # Every (τ, ν) is a t-EV copula, so no joint constraint; the box is the
    # validated range of the fit (the constructor accepts 0.05 ≤ ν ≤ 1e4).
    "nu":    (0.5,   100.0,  4.0),
    # BB6 (FR-9): the outer-power exponent δ ≥ 1, under its own name — not
    # BB1's ``delta``, whose ``_two_parameter_spec`` branch carries BB1's own
    # ``τ = 1 − 2/(δ(θ+2))`` map (``pmcprg.copulas.archimedean.bb6``,
    # "Parametrisation"), exactly as t-EV's ``nu`` is not Student's ``df``.
    # Jointly constrained with τ (δ ≤ 1/(1 − τ)); the init 1.0 is the Joe
    # member, admissible at every registered τ, as Tawn's ψ = 1 is, and the
    # upper bound 10 is admissible for τ ≥ 0.9 only, as BB1's δ = 10 is.
    "delta6": (1.0,   10.0,  1.0),
    # BB7 (FR-9): the **Joe-side exponent θ ≥ 1**, under its own name — not
    # BB1's ``delta`` and not BB6's ``delta6``, since ``_two_parameter_spec``
    # dispatches on the name and each of the three carries a different τ map
    # (``pmcprg.copulas.archimedean.bb7``, "Parametrisation"). θ and not BB7's
    # own δ, because τ is monotone in δ at fixed θ but **not** in θ at fixed δ
    # (it dips below the Clayton value for δ ≳ 3.44), so only (τ, θ) is a
    # one-to-one parametrisation. Jointly constrained with τ
    # (τ > τ_Joe(θ), i.e. θ < θ_Joe(τ)); the init 1.0 is the Clayton member,
    # admissible at every registered τ as BB6's δ = 1 is, and the upper bound
    # 10 is admissible for τ > τ_Joe(10) ≈ 0.816 only, as BB1's δ = 10 is for
    # τ > 0.9.
    "theta7": (1.0,   10.0,  1.0),
    # Three-parameter Tawn (FR-9, round 5): the two weights of the full
    # asymmetric-logistic model, same box as ``psi`` (each is a weight in
    # (0, 1] and the init 1.0 is the Gumbel corner, admissible at every
    # τ < 1). They are *jointly* constrained with τ — τ < 1/(1/ψ_u + 1/ψ_v − 1)
    # — which no box expresses, hence ``CopulaTawn3.constructible_params``.
    # Distinct names from ``psi``: ``_two_parameter_spec`` dispatches on the
    # names, and a family's extras must be distinguishable one from another.
    "psi_u": (0.01,    1.0,  1.0),
    "psi_v": (0.01,    1.0,  1.0),
    # BB8 (FR-9): δ ∈ (0, 1], under its own name — not BB1's ``delta`` nor
    # BB6's ``delta6``, whose ``_two_parameter_spec`` branches carry those
    # families' own τ maps (``pmcprg.copulas.archimedean.bb8``,
    # "Parametrisation"). Unlike every other entry above, this box is the
    # whole admissible set: BB8 puts **no** joint constraint on (τ, δ), every
    # τ ∈ (0, 1) being reached at every δ. The init 1.0 is the Joe member and
    # the lower bound 0.01 is admissible at every τ as well; it is a
    # *fitting* floor (δ → 0 is independence, where δ stops being
    # identifiable), not an admissibility one — the constructor accepts any
    # δ > 0.
    "delta8": (0.01,   1.0,  1.0),
}

# τ-bound padding for the bounded optimisers (``fit(method='mle')`` here,
# ``pmcprg.pmc.ice``) and for pulling a moment estimate off a singular endpoint
# (``fit(method='tau')``). Single source of both constants.
TAU_PAD_REL: float = 1e-4   # relative to the family's τ-span
TAU_PAD_ABS: float = 1e-8   # absolute floor


def is_singular_tau(tau: float) -> bool:
    """``True`` at |τ| = 1 — the Fréchet–Hoeffding bounds.

    Kendall's τ equals 1 only for the comonotone copula M and −1 only for the
    countermonotone copula W (Nelsen 2006, ch. 2 and ch. 5). Both concentrate
    their mass on a diagonal of the unit square: they are singular and have
    no density, so no family can be evaluated there — the closed-form
    parameter maps divide by zero (θ = 2τ/(1 − τ), θ = 1/(1 − τ)) or build a
    singular correlation matrix (ρ = sin(πτ/2) = ±1).
    """
    return bool(abs(float(tau)) >= 1.0)


def padded_tau_range(tau_min: float, tau_max: float) -> tuple[float, float]:
    """``(lo, hi)`` — the τ-range pulled inwards at **both** ends.

    Used by the bounded likelihood optimisers, which must not evaluate a
    density at the edge of its range. A degenerate range (``τ_max == τ_min``,
    the Product copula) is returned untouched: padding it would invert it.
    """
    span = tau_max - tau_min
    if span <= 2.0 * TAU_PAD_ABS:
        return tau_min, tau_max
    pad = max(TAU_PAD_REL * span, TAU_PAD_ABS)
    return tau_min + pad, tau_max - pad


def constructible_tau_range(tau_min: float, tau_max: float) -> tuple[float, float]:
    """``(lo, hi)`` — the registered τ-range with its **singular** ends pulled in.

    Only an endpoint at |τ| = 1 (see :func:`is_singular_tau`) is moved, by the
    same padding as :func:`padded_tau_range`; an admissible endpoint — the
    independence end ε of Clayton/GH/Joe, 1/3 for A12/A14, ±2/9 for FGM —
    is kept, because the family is defined there.
    """
    span = tau_max - tau_min
    pad = max(TAU_PAD_REL * span, TAU_PAD_ABS)
    lo = tau_min + pad if is_singular_tau(tau_min) else tau_min
    hi = tau_max - pad if is_singular_tau(tau_max) else tau_max
    return lo, hi


@dataclass
class CopulaDataMixin:
    ID: int
    SHORT_NAME: str
    LONG_NAME: str
    CLASS_NAME: str
    AVAILABLE: bool
    PARAMETERS_SET_NAME: list[str] = field(default_factory=list)
    TAU_MIN_MAX: list[float] = field(default_factory=list)
    MODULE: str = ""   # dotted import path, e.g. "pmcprg.copulas.elliptical.gaussian"


@unique
class CopulaEnum(CopulaDataMixin, Enum):
    PRODUCT  = 1,  "Prod",    "Product",              "CopulaProduct",  True,  ["tau_k"], [0.0, 0.0],                    "pmcprg.copulas.explicit.product"
    GAUSSIAN = 2,  "Gauss",   "Gaussian",             "CopulaGaussian", True,  ["tau_k"], [-1.0, 1.0],                   "pmcprg.copulas.elliptical.gaussian"
    STUDENT  = 3,  "Student", "Student",              "CopulaStudent",  True,  ["tau_k", "df"], [-1.0, 1.0],             "pmcprg.copulas.elliptical.student"
    GH       = 4,  "GH",      "Gumbel-Hougaard",      "CopulaGH",       True,  ["tau_k"], [0.0 + EPS, 1.0],             "pmcprg.copulas.archimedean.gumbel"
    FGM      = 5,  "FGM",     "Farlie-Gumbel-Morgenstern", "CopulaFGM", True,  ["tau_k"], [-2.0 / 9.0, 2.0 / 9.0],     "pmcprg.copulas.explicit.fgm"
    CUBSEC   = 6,  "CubSec",  "Cubic Section",        "CopulaCubSec",   True,  ["tau_k"], [0.0, 33.0 / 200.0],          "pmcprg.copulas.explicit.cubic_section"
    CLAYTON  = 7,  "Clayton", "Clayton",              "CopulaClayton",  True,  ["tau_k"], [0.0 + EPS, 1.0],             "pmcprg.copulas.archimedean.clayton"
    A12      = 8,  "A12",     "Archimedean12",        "CopulaA12",      True,  ["tau_k"], [1.0 / 3.0, 1.0],             "pmcprg.copulas.archimedean.a12"
    A14      = 9,  "A14",     "Archimedean14",        "CopulaA14",      True,  ["tau_k"], [1.0 / 3.0, 1.0],             "pmcprg.copulas.archimedean.a14"
    FRANK           = 10, "Frank",    "Frank",                    "CopulaFrank",    True,  ["tau_k"], [EPS_MINUS_ONE, ONE_MINUS_EPS],  "pmcprg.copulas.archimedean.frank"
    JOE             = 11, "Joe",      "Joe",                      "CopulaJoe",      True,  ["tau_k"], [0.0 + EPS, 1.0],              "pmcprg.copulas.archimedean.joe"
    SURVIVAL_CLAYTON = 12, "SClayton", "Survival Clayton",        "SurvivalClayton", True, ["tau_k"], [0.0 + EPS, 1.0],             "pmcprg.copulas.archimedean.survival"
    SURVIVAL_GH      = 13, "SGH",      "Survival Gumbel-Hougaard", "SurvivalGH",    True,  ["tau_k"], [0.0 + EPS, 1.0],             "pmcprg.copulas.archimedean.survival"
    SURVIVAL_JOE     = 14, "SJoe",     "Survival Joe",            "SurvivalJoe",    True,  ["tau_k"],          [0.0 + EPS, 1.0],  "pmcprg.copulas.archimedean.survival"
    BB1              = 15, "BB1",      "BB1 (Clayton-Gumbel)",       "CopulaBB1",      True,  ["tau_k", "delta"], [0.0 + EPS, 1.0],              "pmcprg.copulas.archimedean.bb1"
    AMH              = 16, "AMH",      "Ali-Mikhail-Haq",         "CopulaAMH",      True,  ["tau_k"],          [_AMH_TAU_MIN, _AMH_TAU_MAX],  "pmcprg.copulas.archimedean.amh"
    PLACKETT         = 17, "Plackett", "Plackett",                "CopulaPlackett", True,  ["tau_k"],          [EPS_MINUS_ONE, ONE_MINUS_EPS], "pmcprg.copulas.explicit.plackett"
    GALAMBOS         = 18, "Galambos", "Galambos",                "CopulaGalambos", True,  ["tau_k"],          [0.0 + EPS, 1.0],              "pmcprg.copulas.extreme_value.galambos"
    HUSLER_REISS     = 19, "HuslerReiss", "Hüsler-Reiss",          "CopulaHuslerReiss", True, ["tau_k"],        [0.0 + EPS, 1.0],              "pmcprg.copulas.extreme_value.husler_reiss"
    CLAYTON90        = 20, "Clayton90", "Clayton (90° rotation)",  "CopulaClayton90",   True, ["tau_k"],        [-1.0, 0.0 - EPS],             "pmcprg.copulas.archimedean.rotated"
    CLAYTON270       = 21, "Clayton270", "Clayton (270° rotation)", "CopulaClayton270", True, ["tau_k"],        [-1.0, 0.0 - EPS],             "pmcprg.copulas.archimedean.rotated"
    GH90             = 22, "GH90", "Gumbel-Hougaard (90° rotation)",  "CopulaGH90",     True, ["tau_k"],        [-1.0, 0.0 - EPS],             "pmcprg.copulas.archimedean.rotated"
    GH270            = 23, "GH270", "Gumbel-Hougaard (270° rotation)", "CopulaGH270",   True, ["tau_k"],        [-1.0, 0.0 - EPS],             "pmcprg.copulas.archimedean.rotated"
    JOE90            = 24, "Joe90", "Joe (90° rotation)",              "CopulaJoe90",   True, ["tau_k"],        [-1.0, 0.0 - EPS],             "pmcprg.copulas.archimedean.rotated"
    JOE270           = 25, "Joe270", "Joe (270° rotation)",            "CopulaJoe270",  True, ["tau_k"],        [-1.0, 0.0 - EPS],             "pmcprg.copulas.archimedean.rotated"
    BB190            = 26, "BB190", "BB1 (90° rotation)",              "CopulaBB190",   True, ["tau_k", "delta"], [-1.0, 0.0 - EPS],          "pmcprg.copulas.archimedean.rotated"
    BB1270           = 27, "BB1270", "BB1 (270° rotation)",            "CopulaBB1270",  True, ["tau_k", "delta"], [-1.0, 0.0 - EPS],          "pmcprg.copulas.archimedean.rotated"
    TAWN1            = 28, "Tawn1",  "Tawn type 1",                    "CopulaTawn1",   True, ["tau_k", "psi"],   [0.0 + EPS, 1.0],           "pmcprg.copulas.extreme_value.tawn"
    TAWN2            = 29, "Tawn2",  "Tawn type 2",                    "CopulaTawn2",   True, ["tau_k", "psi"],   [0.0 + EPS, 1.0],           "pmcprg.copulas.extreme_value.tawn"
    TEV              = 30, "tEV",    "t extreme-value (t-EV)",         "CopulaTEV",     True, ["tau_k", "nu"],    [0.0 + EPS, 1.0],           "pmcprg.copulas.extreme_value.t_ev"
    # FR-8, last round: mirror images of A12/A14's [1/3, 1] — the first
    # rotated ranges that do not touch τ = 0 (independence is not a member).
    A1290            = 31, "A1290",  "Archimedean12 (90° rotation)",   "CopulaA1290",   True, ["tau_k"],          [-1.0, -1.0 / 3.0],         "pmcprg.copulas.archimedean.rotated"
    A12270           = 32, "A12270", "Archimedean12 (270° rotation)",  "CopulaA12270",  True, ["tau_k"],          [-1.0, -1.0 / 3.0],         "pmcprg.copulas.archimedean.rotated"
    A1490            = 33, "A1490",  "Archimedean14 (90° rotation)",   "CopulaA1490",   True, ["tau_k"],          [-1.0, -1.0 / 3.0],         "pmcprg.copulas.archimedean.rotated"
    A14270           = 34, "A14270", "Archimedean14 (270° rotation)",  "CopulaA14270",  True, ["tau_k"],          [-1.0, -1.0 / 3.0],         "pmcprg.copulas.archimedean.rotated"
    # FR-8, closing round: the audit's own family list also names "BB1 de
    # survie" (survival BB1) — the 180° rotation of BB1, missing until now —
    # and its 90°/270° children. τ is unchanged by a 180° rotation, so
    # SURVIVAL_BB1's range mirrors BB1's own [0+ε, 1); the two rotations then
    # mirror BB190/BB1270's own [-1, -ε] exactly like every other 90°/270°
    # pair here.
    SURVIVAL_BB1     = 35, "SBB1",    "Survival BB1 (Clayton-Gumbel)",         "SurvivalBB1",    True, ["tau_k", "delta"], [0.0 + EPS, 1.0],   "pmcprg.copulas.archimedean.survival"
    SURVIVAL_BB190   = 36, "SBB190",  "Survival BB1 (90° rotation)",        "SurvivalBB190",  True, ["tau_k", "delta"], [-1.0, 0.0 - EPS],  "pmcprg.copulas.archimedean.rotated"
    SURVIVAL_BB1270  = 37, "SBB1270", "Survival BB1 (270° rotation)",       "SurvivalBB1270", True, ["tau_k", "delta"], [-1.0, 0.0 - EPS],  "pmcprg.copulas.archimedean.rotated"
    # FR-9: BB6, the outer power (Gumbel transform) of Joe — Joe at δ = 1,
    # Gumbel–Hougaard at θ = 1, λ_U = 2 − 2^{1/(θδ)} and λ_L = 0. Its second
    # parameter is registered as ``delta6``, not ``delta``, so the joint
    # fitter cannot route it through BB1's branch
    # (``pmcprg.copulas.archimedean.bb6`` module docstring). The τ-range is
    # Joe's own: every τ ∈ (0, 1) is reached, at δ = 1.
    BB6              = 38, "BB6",     "BB6 (Joe-Gumbel)",                   "CopulaBB6",      True, ["tau_k", "delta6"], [0.0 + EPS, 1.0],  "pmcprg.copulas.archimedean.bb6"
    # FR-9, round 5: the full asymmetric-logistic Tawn model, both weights
    # free — the FIRST family here with three parameters (τ + two extras).
    # Its τ-range is Tawn 1/2's own [0+ε, 1]; the pair (ψ_u, ψ_v) narrows the
    # *reachable* τ jointly (``CopulaTawn3.reachable_tau_cap``), not the
    # registered range, exactly as BB1's δ does.
    TAWN3            = 39, "Tawn3",   "Tawn (asymmetric logistic, 3 par.)", "CopulaTawn3",    True, ["tau_k", "psi_u", "psi_v"], [0.0 + EPS, 1.0], "pmcprg.copulas.extreme_value.tawn"
    # FR-9, round 6: BB7, the Joe-Clayton copula — Clayton at θ = 1, Joe as
    # δ → 0, λ_U = 2 − 2^{1/θ} and λ_L = 2^{−1/δ}, the first family here with
    # a *two-way* nest and one tail coefficient per parameter. Its second
    # parameter is registered as ``theta7`` — θ, not δ, because τ is *not*
    # monotone in θ at fixed δ — and under its own name, so the joint fitter
    # cannot route it through BB1's or BB6's branch
    # (``pmcprg.copulas.archimedean.bb7`` module docstring). The τ-range is
    # Joe's own: every τ ∈ (0, 1) is reached, at θ = 1 (Clayton).
    BB7              = 40, "BB7",     "BB7 (Joe-Clayton)",                  "CopulaBB7",      True, ["tau_k", "theta7"], [0.0 + EPS, 1.0],  "pmcprg.copulas.archimedean.bb7"
    # FR-9, BB8 round: the last of the BB families the audit names. Joe at
    # δ = 1, the independence copula at θ = 1 (*not* Frank, despite the
    # "Joe-Frank" nickname — Frank is only the joint limit θ → ∞, δ → 0 with
    # θδ fixed), and **no upper-tail dependence at all for δ < 1**. Its second
    # parameter is registered as ``delta8`` — neither BB1's ``delta`` nor
    # BB6's ``delta6`` — so the joint fitter cannot route it through another
    # family's branch (``pmcprg.copulas.archimedean.bb8`` module docstring).
    # The τ-range is Joe's own and, unlike BB1/BB6/Tawn, is reached *whole* at
    # every δ: BB8 has no joint (τ, δ) constraint. τ has no elementary closed
    # form here (a ₃F₂ one, unusable numerically) and is computed by
    # quadrature — the first Archimedean family in the package that needs one.
    # ID reconciled with BB7 (both agents worked from the same 40-slot base):
    # BB7 kept 40 (integrated first), BB8 takes the next free slot, 41.
    BB8              = 41, "BB8",     "BB8 (Joe-Frank)",                    "CopulaBB8",      True, ["tau_k", "delta8"], [0.0 + EPS, 1.0],  "pmcprg.copulas.archimedean.bb8"

    def describe(self):
        return self.name, self.value

    def __reduce_ex__(self, protocol):
        """Pickle a member by its **name**, not by its value.

        ``Enum`` pickles by value, i.e. it stores the whole registry entry —
        ID, both names, τ-range, parameter names *and the dotted MODULE path*
        — and looks the member up by equality on load. Any later edit of one
        of those fields (a τ-range fixed, a parameter added, a module moved:
        this registry has seen all three) made every previously pickled copula
        of that family unloadable. The name (``"BB1"``, ``"TAWN3"``…) is the
        stable identity: it is what TOML model files, ``CopulaEnum[...]``
        lookups and the registry snapshot test already rely on.

        A pickle written by an earlier version, which stores the value,
        still loads through ``CopulaEnum(value)`` as long as that entry has
        not changed since.
        """
        return getattr, (type(self), self._name_)

    def constructible_tau_range(self) -> tuple[float, float]:
        """The registered τ-range without its singular endpoints (|τ| = 1),
        cut to the τ the family's parameter map reaches.

        See :func:`constructible_tau_range`. Every τ in this closed interval
        passes the base-class check of :class:`CopulaVirt` and is stored
        unchanged by the copula. The second cut only applies to a family that
        declares :meth:`CopulaVirt.reachable_tau_bounds` (Frank ±0.994299,
        Plackett [−0.993525, 0.993525]): a τ beyond builds the copula at the
        capped parameter and stores the τ of that parameter, so clipping into
        this range never produces a τ the copula does not use.
        """
        lo, hi = constructible_tau_range(*self.value.TAU_MIN_MAX)
        bounds = self.klass.reachable_tau_bounds()
        if bounds is not None:
            lo, hi = max(lo, bounds[0]), min(hi, bounds[1])
        return lo, hi

    def reachable_tau(self, tau):
        """``tau``, or the nearest τ the family represents when ``tau`` is beyond it.

        ``tau`` itself is returned — unchanged, same object — for a family
        without :meth:`CopulaVirt.reachable_tau_bounds`, for every τ within
        the bounds and for NaN. Beyond them (Frank: |τ| > 0.994299, θ = ±700;
        Plackett: beyond its τ table, θ = 10^{±6}) the copula already uses the
        parameter of the bound, so its density is the same at ``tau`` and at
        the returned value: an optimiser may evaluate the returned value
        without changing its objective (and without the family's clamping
        WARNING), and a fitted τ is stored as the τ the copula uses.
        """
        bounds = self.klass.reachable_tau_bounds()
        if bounds is None:
            return tau
        lo, hi = bounds
        if tau > hi:
            return hi
        if tau < lo:
            return lo
        return tau

    def correct_tau(self, tau: float, *, warn: bool = True) -> float:
        """Clip tau to the registered range; replace non-finite values with the midpoint.

        Logs a ``WARNING`` when the input is outside the family's valid τ
        range or non-finite, so that callers (e.g. ICE candidate selection)
        notice the silent clipping. ``warn=False`` logs the same message at
        ``DEBUG`` instead — for a τ the package drew itself (multistart
        jitter), which is no input error of the user; the result is the same.

        The target is the *registered* range, whose endpoints may be singular
        (|τ| = 1 is refused by the constructor, RB-6). A caller that must be
        able to build the copula clips further into
        :meth:`constructible_tau_range`.
        """
        log = logger.warning if warn else logger.debug
        tau_min, tau_max = self.value.TAU_MIN_MAX
        if not np.isfinite(tau):
            mid = 0.5 * (tau_min + tau_max)
            log(
                "%s.correct_tau: τ=%r is non-finite — using midpoint τ=%.4f.",
                self.value.SHORT_NAME, tau, mid,
            )
            return mid
        if tau < tau_min or tau > tau_max:
            clipped = float(np.clip(tau, tau_min, tau_max))
            log(
                "%s.correct_tau: τ=%.4f outside valid range [%.4f, %.4f] — "
                "clipped to τ=%.4f.",
                self.value.SHORT_NAME, tau, tau_min, tau_max, clipped,
            )
            return clipped
        return float(tau)

    @classmethod
    def favorite(cls):
        return cls.GAUSSIAN

    @classmethod
    def print_available(cls):
        logger.info("Available copulas:")
        for c in CopulaEnum:
            if c.value.AVAILABLE:
                logger.info("  %s", c.describe())

    @classmethod
    def available(cls):
        return [c for c in CopulaEnum if c.value.AVAILABLE]

    @classmethod
    def available_class_names(cls):
        return [c.value.CLASS_NAME for c in CopulaEnum if c.value.AVAILABLE]

    @classmethod
    def from_short_name(cls, short: str):
        for c in CopulaEnum:
            if c.value.SHORT_NAME == short:
                return c
        return None

    @property
    def klass(self):
        """The implementation class for this copula, imported lazily and cached.

        Avoids re-running ``importlib.import_module`` on every model build /
        ICE candidate evaluation. Keyed by the (immutable) ``CLASS_NAME``
        because ``CopulaEnum`` instances are not hashable (the dataclass
        mixin overrides ``__eq__``).
        """
        key = self.value.CLASS_NAME
        cls_obj = _COPULA_KLASS_CACHE.get(key)
        if cls_obj is None:
            import importlib as _importlib
            mod = _importlib.import_module(self.value.MODULE)
            cls_obj = getattr(mod, self.value.CLASS_NAME)
            _COPULA_KLASS_CACHE[key] = cls_obj
        return cls_obj


class CopulaVirt:

    # Number of free parameters; subclasses override if they fit more than τ_K
    n_params: int = 1

    # Largest |τ| the family's parameter map reaches when it is smaller than
    # the registered range and symmetric; ``None`` — no such bound. A τ beyond
    # builds the copula at the capped parameter, which then stores the τ of
    # that parameter. The default reachable_tau_bounds() reads it.
    # Frank: θ ≤ 700, |τ| ≤ 0.994299.
    reachable_tau_abs: float | None = None

    @classmethod
    def reachable_tau_bounds(cls) -> tuple[float, float] | None:
        """``(lo, hi)`` — the τ the family's parameter map reaches, when narrower
        than the registered range; ``None`` when every registered τ is reached.

        A τ beyond builds the copula at the capped parameter, which then stores
        the τ of that parameter. Read by :meth:`CopulaEnum.constructible_tau_range`
        and :meth:`CopulaEnum.reachable_tau`. Default: ``±reachable_tau_abs``
        (Frank). A family whose bounds are not opposite or not known at import
        overrides it — Plackett, whose τ table is computed on first use and ends
        at −0.9935245713002141 / 0.9935245713002134.
        """
        cap = cls.reachable_tau_abs
        return None if cap is None else (-cap, cap)

    def __init__(self, class_name: str, params: dict):
        self.class_name = class_name

        self.copula_enum = None
        for c in CopulaEnum.available():
            if c.value.CLASS_NAME == self.class_name:
                self.copula_enum = c
                self.tau_min = c.value.TAU_MIN_MAX[0]
                self.tau_max = c.value.TAU_MIN_MAX[1]
                break
        if self.copula_enum is None:
            raise CopulaNotAvailableError(
                f"Copula {self.class_name!r} is not available."
            )

        for k in params:
            if k not in self.copula_enum.value.PARAMETERS_SET_NAME:
                raise CopulaParameterError(
                    f"Unexpected parameter {k!r} for {self.class_name}. "
                    f"Expected: {self.copula_enum.value.PARAMETERS_SET_NAME}"
                )

        if "tau_k" not in params:
            raise CopulaParameterError(
                f"Missing required parameter tau_k for {self.class_name}."
            )

        self._check_tau_k(params["tau_k"])

        self.params = params

        self._update_params()

        # Grid for the numerical majorant (150 points on (0,1))
        self.N = 150
        self.ticks_nbr = 15
        self._init_grids()

    #: Attributes built by :meth:`_init_grids`: pure functions of ``N``, so
    #: left out of a pickle and rebuilt on load (see :meth:`__getstate__`).
    _GRID_ATTRS = ("_x", "_y", "_x1", "_y1", "_x2", "_y2", "_z2")

    def _init_grids(self) -> None:
        """The numerical-majorant grid and the plotting mesh, from ``self.N``."""
        # Grid for the numerical majorant (N points on (0,1))
        self._x = np.linspace(EPS, ONE_MINUS_EPS, self.N)
        self._y = np.zeros(self.N)

        # Grid for plotting (N×N)
        self._x1 = np.linspace(EPS, ONE_MINUS_EPS, self.N)
        self._y1 = np.linspace(EPS, ONE_MINUS_EPS, self.N)
        self._x2, self._y2 = np.meshgrid(self._x1, self._y1)
        self._z2 = np.zeros(self._x2.shape)

    def __getstate__(self) -> dict:
        """The instance state without the rebuildable grids.

        Three of them are N×N = 150×150 float64 arrays (180 kB each): a
        pickled copula was 0.5 MB before any parameter, 1.1–1.6 MB for a
        rotated or survival one (which holds a base copula), and a two-state
        PMC model 2.2 MB — copied into every worker of a multistart and into
        every ``copy.deepcopy``. They are deterministic in ``N``, so
        :meth:`__setstate__` rebuilds them bit for bit.
        """
        state = self.__dict__.copy()
        for name in self._GRID_ATTRS:
            state.pop(name, None)
        return state

    def __setstate__(self, state: dict) -> None:
        # Also accepts the state of a pickle written before the grids were
        # dropped: those keys are simply overwritten by the identical rebuild.
        self.__dict__.update(state)
        self._init_grids()

    # ------------------------------------------------------------------
    # Internal parameter update (must be overridden by every subclass)
    # ------------------------------------------------------------------
    def _update_params(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _update_params()."
        )

    def _check_tau_k(self, tau) -> None:
        """Refuse a τ the family cannot be built at — before ``_update_params`` runs.

        Raises :class:`CopulaParameterError` when τ is not a finite number,
        lies outside the registered range, or sits at a singular endpoint
        |τ| = 1 (audit RB-6). The last check used to be left to each family,
        whose parameter map then raised ``ZeroDivisionError`` (GH, Clayton,
        A12, A14 and their survivals), ``LinAlgError`` (Gaussian, Student) or
        returned θ = 10⁹ (Joe) at the registered bound τ = 1.
        """
        if isinstance(tau, (bool, np.bool_)) or not isinstance(tau, numbers.Real):
            raise CopulaParameterError(
                f"tau_k={tau!r} is not a real number for {self.class_name}."
            )
        tau = float(tau)
        if not np.isfinite(tau) or not (self.tau_min <= tau <= self.tau_max):
            raise CopulaParameterError(
                f"tau_k={tau} out of [{self.tau_min}, {self.tau_max}] "
                f"for {self.class_name}."
            )
        if is_singular_tau(tau):
            bound = "comonotone copula M" if tau > 0 else "countermonotone copula W"
            lo, hi = constructible_tau_range(self.tau_min, self.tau_max)
            raise CopulaParameterError(
                f"{self.class_name}: tau_k={tau:g} is the Fréchet–Hoeffding "
                f"bound ({bound}), a singular copula without a density — the "
                f"family is only defined strictly inside it. Use τ in "
                f"[{lo:.6g}, {hi:.6g}]."
            )

    # ------------------------------------------------------------------
    # Admissible-set projection (overridden by jointly-constrained families)
    # ------------------------------------------------------------------
    @classmethod
    def constrain_params(cls, params: dict) -> dict:
        """Project ``params`` onto the family's admissible set.

        Identity for every family whose parameters are independently bounded.
        Families with a *joint* constraint override this — BB1 needs
        ``δ < 1/(1−τ)`` for ``θ > 0``, which a box cannot express. Bounded
        optimisers only accept per-parameter intervals, so without this hook
        they evaluate inadmissible points, get a constant failure penalty, and
        stall on that plateau with a zero gradient.
        """
        return dict(params)

    @classmethod
    def constructible_params(cls, params: dict) -> dict:
        """``params`` itself when the constructor accepts it, else the nearest point it accepts.

        ``params`` holds ``tau_k`` and any extra parameters (a TOML block's
        values under the constructor's names). The multistart draws move τ
        and each extra parameter within its own box
        (:meth:`CopulaEnum.constructible_tau_range`,
        ``EXTRA_PARAM_BOUNDS_BY_PARAM``); a family whose parameters are
        *jointly* constrained overrides this hook to repair a draw that
        leaves the admissible set — BB1: ``δ < 1/(1 − τ)``. Unlike
        :meth:`constrain_params`, which projects onto an optimiser's fitting
        box, nothing is changed at a point the constructor accepts: the same
        dict is returned, so a draw that builds is kept bit for bit.
        Identity for every family with independent parameter bounds.
        """
        return params

    # ------------------------------------------------------------------
    # PDF / CDF (must be overridden by every subclass)
    # ------------------------------------------------------------------
    def pdf(self, uv):
        raise NotImplementedError(f"{self.__class__.__name__} must implement pdf().")

    def cdf(self, uv):
        raise NotImplementedError(f"{self.__class__.__name__} must implement cdf().")

    # ------------------------------------------------------------------
    # Vectorised PDF — used by hot loops in pmcprg.pmc (forward-backward, ICE).
    #
    # The two array densities are defaults for each other, and neither floors
    # (audit FR-2 / RB-9):
    #   * a subclass with a native ``logpdf_array`` gets
    #     ``pdf_array = exp(logpdf_array)`` — the two can then never disagree;
    #   * otherwise ``pdf_array`` comes from the backend (below) and
    #     ``logpdf_array = log(pdf_array)``, −∞ where the density is 0.0.
    # The backend is an optional vectorised object in ``self._model`` (any
    # object whose ``.pdf``/``.cdf`` accept an (M, 2) ndarray — historically a
    # statsmodels copula; no family uses one any more and statsmodels is not a
    # dependency), else a scalar loop over ``pdf``.
    # ------------------------------------------------------------------
    def _overrides(self, name: str) -> bool:
        """``True`` when the subclass redefines the method ``name``."""
        return getattr(type(self), name) is not getattr(CopulaVirt, name)

    def _backend_pdf_array(self, uv: np.ndarray) -> np.ndarray:
        """c(u, v) from the optional vectorised ``self._model`` or the scalar ``pdf`` loop.

        No floor: an underflowed density is returned as 0.0 and a NaN from the
        backend as NaN. Only a negative value — a rounding residue of a
        density that is ≥ 0 by definition — is set to 0.0.
        """
        uv = np.asarray(uv, dtype=float)
        if uv.ndim != 2 or uv.shape[1] != 2:
            raise ValueError(f"pdf_array expects shape (M, 2); got {uv.shape}.")

        # Clamp inputs to (EPS, 1-EPS) to mirror scalar minmaxEPS
        uv = np.clip(uv, EPS, ONE_MINUS_EPS)

        if hasattr(self, "_model") and hasattr(self._model, "pdf"):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                vals = np.array(self._model.pdf(uv), dtype=float, copy=True).reshape(-1)
        else:
            # Scalar loop (slow). Subclasses without ``_model`` should
            # override pdf_array or logpdf_array with a vectorised closed form.
            vals = np.empty(uv.shape[0], dtype=float)
            for k in range(uv.shape[0]):
                vals[k] = self.pdf([float(uv[k, 0]), float(uv[k, 1])])
        vals[vals < 0.0] = 0.0
        return vals

    def pdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Evaluate c(u, v) on M point pairs.

        Parameters
        ----------
        uv : np.ndarray, shape (M, 2)

        Returns
        -------
        np.ndarray, shape (M,) — copula density values, **not floored**: 0.0
        where the density underflows. ``exp(logpdf_array)`` when the subclass
        provides a native log-density, the backend values otherwise.
        """
        if self._overrides("logpdf_array"):
            uv = np.asarray(uv, dtype=float)
            if uv.ndim != 2 or uv.shape[1] != 2:
                raise ValueError(f"pdf_array expects shape (M, 2); got {uv.shape}.")
            with np.errstate(over="ignore"):
                return np.exp(self.logpdf_array(uv))
        return self._backend_pdf_array(uv)

    def logpdf_array(self, uv: np.ndarray) -> np.ndarray:
        """log c(u, v) on M point pairs — ``log(pdf_array)``, **not floored**.

        The former default ``log(max(pdf, EPS))`` capped every log-density at
        log EPS = −36.04 nat, flattening the likelihood surface wherever the
        true values are lower (−289 to −3086 nat for Student at τ = 0.95,
        ν = 30; audit RB-4). An exact zero now gives −∞, which the likelihood
        objectives handle (:func:`pmcprg.copulas._fit._weighted_log_density_sum`).
        """
        pdf = self.pdf_array(uv) if self._overrides("pdf_array") else self._backend_pdf_array(uv)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(pdf)

    def cdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Evaluate C(u, v) on M point pairs.

        Same fast-path strategy as :meth:`pdf_array`: use the optional
        vectorised backend ``self._model`` if present (its ``.cdf`` accepts a
        (M, 2) ndarray and returns an (M,) ndarray), otherwise fall back to a
        Python loop.

        Subclasses with closed-form CDFs (FGM, AMH, Joe, …) can override this
        with a vectorised expression for ~50× speed-ups on bootstrap GoF.

        Parameters
        ----------
        uv : np.ndarray, shape (M, 2)

        Returns
        -------
        np.ndarray, shape (M,) — clipped to [0, 1]; NaN where the backend
        returns a non-finite value.

        A non-finite backend value used to be replaced by EPS, a plausible
        number in the lower corner and a wrong one wherever C is close to 1
        (audit RB-9). NaN makes the failure visible to the caller instead.
        """
        uv = np.asarray(uv, dtype=float)
        if uv.ndim != 2 or uv.shape[1] != 2:
            raise ValueError(f"cdf_array expects shape (M, 2); got {uv.shape}.")

        uv = np.clip(uv, EPS, ONE_MINUS_EPS)

        if hasattr(self, "_model") and hasattr(self._model, "cdf"):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                vals = np.asarray(self._model.cdf(uv), dtype=float)
            vals = np.where(np.isfinite(vals), vals, np.nan)
            return np.clip(vals, 0.0, 1.0)

        # Fallback — scalar loop (slow). Subclasses without ``_model`` should
        # override cdf_array with a vectorised closed form.
        out = np.empty(uv.shape[0], dtype=float)
        for k in range(uv.shape[0]):
            out[k] = self.cdf([float(uv[k, 0]), float(uv[k, 1])])
        return np.clip(out, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Conditional CDF  h(v|u) = ∂C(u,v)/∂u
    # ------------------------------------------------------------------
    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = ∂C(u,v)/∂u via central finite differences. Subclasses override analytically."""
        h = 1e-5
        u_lo = minmaxEPS(u - h)
        u_hi = minmaxEPS(u + h)
        result = (self.cdf([u_hi, v]) - self.cdf([u_lo, v])) / (u_hi - u_lo)
        return float(np.clip(result, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Transpose — conditioning on the second argument (audit FR-11)
    # ------------------------------------------------------------------
    #: ``C(u, v) = C(v, u)``. True for every family except the 90°/270°
    #: rotations and the Tawn models, which override :meth:`transposed`.
    exchangeable: bool = True

    def transposed(self) -> "CopulaVirt":
        """The copula of (V, U) when (U, V) has this one: Cᵀ(u, v) = C(v, u).

        The h-functions condition on the *first* argument; conditioning on
        the second is conditioning on the first of the transpose:

            ∂C/∂v (u, v) = P(U ≤ u | V = v) = ``transposed().conditional_cdf(u, v)``,

        inverted in u by ``transposed().inv_h(w, v)``. ``self`` for an
        exchangeable family. Checked for every registered family
        (``pmcprg/tests/test_copula_transpose.py``), and against the h2 and
        h⁻¹ of pyvinecopulib, VineCopula and R ``copula``
        (``pmcprg/tests/test_parity_interior.py``).
        """
        if self.exchangeable:
            return self
        raise NotImplementedError(
            f"{type(self).__name__} is not exchangeable and defines no transpose."
        )

    # ------------------------------------------------------------------
    # Inverse of the conditional CDF (Rosenblatt sampling step).
    #
    # Default: numerical inversion via Brent's method on h(v|u) = w. Subclasses
    # with an analytical form (e.g. Gaussian, Clayton) override this for a
    # ~50× speed-up on long simulations.
    # ------------------------------------------------------------------
    def inv_h(self, w: float, u: float) -> float:
        """Return v such that h(v|u) = w (Rosenblatt inverse step).

        Default numerical implementation: bracket ``h(v|u) - w`` on
        ``(EPS, 1-EPS)`` and solve via Brent's method. Brent requires the
        function to be monotone in v on that bracket — this holds for any
        valid copula (h is a conditional CDF in v) but the assertion below
        guards against an ill-defined custom copula.

        Subclasses with closed-form inverses (Gaussian, Clayton, Frank, …)
        override this for ~50× speed-up.

        Saturation (kept deliberately, audit RB-9): when ``w`` lies outside
        the range ``[h(EPS|u), h(1−EPS|u)]`` that the bracket can attain, the
        root is outside ``(EPS, 1−EPS)`` and the method returns the nearer
        bracket end, ``EPS`` or ``1−EPS``. This is the value of the exact
        inverse clamped to the grid every sampler in the package works on —
        not a failure — so it is not reported.
        """
        from scipy.optimize import brentq
        u    = minmaxEPS(u)
        h_lo = self.conditional_cdf(EPS,           u)
        h_hi = self.conditional_cdf(ONE_MINUS_EPS, u)

        # h(v|u) is a CDF in v → must be non-decreasing on (0, 1).
        # A violation here indicates a buggy ``conditional_cdf`` override.
        if h_lo > h_hi + 1e-9:
            raise ValueError(
                f"{type(self).__name__}.conditional_cdf is not monotone in v "
                f"at u={u:.4g} (h(EPS|u)={h_lo:.4g} > h(1-EPS|u)={h_hi:.4g}). "
                f"Cannot invert h."
            )

        if w <= h_lo:
            return float(EPS)
        if w >= h_hi:
            return float(ONE_MINUS_EPS)
        return float(brentq(
            lambda v_: self.conditional_cdf(float(v_), u) - w,
            EPS, ONE_MINUS_EPS, maxiter=100, xtol=1e-15,
        ))

    def inv_h_array(self, w: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Vectorised version of :meth:`inv_h`.

        Default fallback: scalar Python loop. Subclasses with closed-form
        ``inv_h`` (Gaussian, Clayton, Frank) override this with a
        vectorised expression for ~10× speed-up on bulk sampling.

        Parameters
        ----------
        w : np.ndarray, shape (M,) — uniform variates.
        u : np.ndarray, shape (M,) — conditioning values.

        Returns
        -------
        np.ndarray, shape (M,) — v values such that h(v_i | u_i) = w_i.
        """
        w = np.asarray(w, dtype=float)
        u = np.asarray(u, dtype=float)
        if w.shape != u.shape:
            raise ValueError(f"inv_h_array: w {w.shape} and u {u.shape} must match.")
        out = np.empty_like(w)
        for k in range(w.size):
            out[k] = self.inv_h(float(w[k]), float(u[k]))
        return out

    # ------------------------------------------------------------------
    # Tail dependence coefficients  (λ_L, λ_U)
    # ------------------------------------------------------------------
    def tail_dependence(self) -> tuple[float, float]:
        """Lower/upper tail dependence (λ_L, λ_U).

        Default numerical implementation via diagonal CDF limits:
            λ_L = lim_{u→0+} C(u,u)/u
            λ_U = lim_{u→1−} (1 − 2u + C(u,u))/(1 − u)
        Subclasses with analytical formulas override this.
        Returns (nan, nan) if the copula has no closed-form CDF.
        """
        try:
            u0, u1 = 1e-4, 1.0 - 1e-4
            c0 = float(self.cdf([u0, u0]))
            c1 = float(self.cdf([u1, u1]))
            lam_L = c0 / u0
            lam_U = (1.0 - 2.0 * u1 + c1) / (1.0 - u1)
            return (float(np.clip(lam_L, 0.0, 1.0)),
                    float(np.clip(lam_U, 0.0, 1.0)))
        except NotImplementedError:
            return float('nan'), float('nan')

    # ------------------------------------------------------------------
    # Majorant  max_{v} c(u_left, v)  — overridden analytically when possible
    # ------------------------------------------------------------------
    def majorant(self, u_left: float) -> float:
        """Compute ``max_v c(u_left, v)`` for the AR-rejection sampler.

        Strategy (most-robust default):

        1. Coarse grid search (150 points) to localise the peak.
        2. Refinement via :func:`scipy.optimize.minimize_scalar` (Brent /
           bounded) in a tight neighbourhood of the grid maximum.
        3. **Safety multiplier ×1.02** to absorb residual numerical error.
           Acceptance rate drops by ≤2 % but the AR sampler is guaranteed
           unbiased.

        Subclasses with closed-form maxima (FGM, CubSec, Product, Gaussian,
        Clayton, …) override this method for ~10× speed-up and exact bounds.
        """
        from scipy.optimize import minimize_scalar
        u = minmaxEPS(u_left)

        # 1. Coarse grid search
        for i, v in enumerate(self._x):
            self._y[i] = self.pdf([u, v])
        idx_grid_max = int(np.argmax(self._y))
        v_grid_max   = self._x[idx_grid_max]
        f_grid_max   = self._y[idx_grid_max]

        # 2. Refinement around the grid max — bracket of width ~2 grid spacings
        spacing  = (ONE_MINUS_EPS - EPS) / max(self.N - 1, 1)
        v_lo     = float(max(EPS,         v_grid_max - 2.0 * spacing))
        v_hi     = float(min(ONE_MINUS_EPS, v_grid_max + 2.0 * spacing))
        try:
            res = minimize_scalar(
                lambda v_: -self.pdf([u, float(v_)]),
                bounds=(v_lo, v_hi),
                method="bounded",
                options={"xatol": 1e-7},
            )
            f_refined = self.pdf([u, float(res.x)])
            f_max     = max(f_grid_max, f_refined)
        except Exception:
            f_max = f_grid_max   # grid is the safety net

        # 3. Safety margin (constant 2 %) — guards against sub-grid peaks the
        #    refinement window may have missed.
        return float(f_max * 1.02)

    # ------------------------------------------------------------------
    # tau update
    # ------------------------------------------------------------------
    def update_tau_k(self, new_tau_k: float):
        """Set τ and refresh every derived parameter (θ, backend model, …).

        Same rule as the constructor: a τ outside the registered range, at a
        singular endpoint |τ| = 1, or not finite raises
        :class:`CopulaParameterError` and leaves the copula unchanged. The
        former behaviour — replace an out-of-range τ by the middle of the
        range *without* calling ``_update_params`` — left θ describing the
        previous τ (audit K-9). If the family's own parameter map refuses the
        new τ, the previous state is restored before the error propagates.
        """
        self._check_tau_k(new_tau_k)
        old_tau = self.params["tau_k"]
        self.params["tau_k"] = new_tau_k
        try:
            self._update_params()
        except Exception:
            self.params["tau_k"] = old_tau
            self._update_params()
            raise

    @property
    def tau_range(self) -> tuple[float, float]:
        return self.tau_min, self.tau_max

    # ------------------------------------------------------------------
    # Parameter estimation
    # ------------------------------------------------------------------
    @classmethod
    def fit(cls, data: np.ndarray, method: str = 'tau') -> 'FitResult':
        """Fit a copula's free parameters from a (n, 2) data array.

        Raw data is rank-transformed to pseudo-observations û = rank/(n+1)
        before estimation — no distributional assumption on the margins.

        Parameters
        ----------
        data   : array-like, shape (n, 2)
        method : {'tau', 'mle'}
            ``'tau'`` — moment matching via Kendall's τ (O(n log n), default):
                       inversion of Kendall's τ, Genest & Rivest (1993).
            ``'mle'`` — maximise ∑ log c(û_i, v̂_i). For 1-parameter families
                       a 1-D Brent scalar search on the padded τ-range; for
                       two-parameter families that do not override ``fit``
                       the joint optimiser of the ICE M-step
                       (:func:`pmcprg.copulas._fit._fit_two_parameter_mle`). On
                       rank pseudo-observations this is the
                       pseudo-maximum-likelihood estimator of Genest, Ghoudi
                       & Rivest (1995).

        With ``'tau'``, τ̂ is clipped into the *constructible* τ-range: a
        singular registered endpoint (|τ| = 1) is replaced by the padded
        value next to it, an admissible endpoint is kept (τ̂ ≤ 0 → ε for
        Clayton/GH/Joe), and a τ̂ beyond the family's reachable |τ| goes to
        that bound (Frank). With ``'mle'``, a τ̂ found beyond the reachable
        |τ| — where the likelihood is flat — is returned as the bound too, so
        ``tau_k`` is always the τ the fitted copula uses.
        ``FitResult.converged`` is ``False`` when an MLE optimiser did not
        converge.

        Returns
        -------
        FitResult

        References
        ----------
        * Genest, C. & Rivest, L.-P. (1993). Statistical inference procedures
          for bivariate Archimedean copulas. *JASA* 88(423), 1034–1043.
        * Genest, C., Ghoudi, K. & Rivest, L.-P. (1995). A semiparametric
          estimation procedure of dependence parameters in multivariate
          families of distributions. *Biometrika* 82(3), 543–552.
        """
        from scipy.stats    import rankdata, kendalltau as _kendalltau
        from scipy.optimize import minimize_scalar

        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError(f'data must be shape (n, 2), got {data.shape}.')
        n = data.shape[0]
        if n < 4:
            raise ValueError(f'At least 4 observations required, got {n}.')

        uv = np.column_stack([
            rankdata(data[:, 0]) / (n + 1),
            rankdata(data[:, 1]) / (n + 1),
        ])

        class_name = cls.__name__
        tau_min = tau_max = None
        param_names: list[str] = []
        for entry in CopulaEnum:
            if entry.value.CLASS_NAME == class_name:
                if not entry.value.AVAILABLE:
                    raise CopulaNotAvailableError(
                        f'{class_name} is currently disabled.')
                tau_min, tau_max = entry.value.TAU_MIN_MAX
                param_names = list(entry.value.PARAMETERS_SET_NAME)
                break
        if tau_min is None:
            raise CopulaNotAvailableError(
                f'{class_name!r} is not registered in CopulaEnum.')

        # Single-point family (Product copula)
        if abs(tau_max - tau_min) < 1e-8:
            tau_k  = tau_min
            copula = cls(tau_k=tau_k)
            return FitResult(copula=copula, method=method, tau_k=tau_k,
                             log_likelihood=_eval_log_likelihood(copula, uv),
                             n_obs=n, uv=uv)

        # Extra (non-tau) parameter bounds — single source of truth shared with
        # ICE-driven fitting (see EXTRA_PARAM_BOUNDS_BY_PARAM at module top).
        extras = {p: EXTRA_PARAM_BOUNDS_BY_PARAM[p] for p in param_names
                  if p != "tau_k" and p in EXTRA_PARAM_BOUNDS_BY_PARAM}
        converged = True

        if method == 'tau':
            tau_hat, _ = _kendalltau(data[:, 0], data[:, 1])
            if not np.isfinite(tau_hat):
                raise ValueError(
                    f"{class_name}.fit(method='tau'): Kendall's τ is undefined "
                    f"on these data (a constant column?)."
                )
            # Clip into the *constructible* range (audit RB-6): a registered
            # endpoint at |τ| = 1 is singular — τ̂ = ±1 on comonotone data used
            # to give LinAlgError (Gaussian), θ ≈ 1e16 with LL = −∞ (Clayton)
            # or absurd likelihoods (GH, A14). An admissible endpoint is kept:
            # τ̂ ≤ 0 still goes to the independence end ε of a one-sided family.
            # The range is also cut to the reachable |τ| (Frank, F1).
            lo, hi = entry.constructible_tau_range()
            tau_k = float(np.clip(tau_hat, lo, hi))
            if tau_k != tau_hat:
                logger.info(
                    "%s.fit(method='tau'): τ̂ = %.6g outside [%.6g, %.6g] — "
                    "clipped to %.6g.", class_name, tau_hat, lo, hi, tau_k,
                )
            extra_vals = {k: v[2] for k, v in extras.items()}   # init defaults

        elif method == 'mle':
            tau_lo, tau_hi = padded_tau_range(tau_min, tau_max)

            if not extras:
                # 1-D scalar optimisation over τ. A NaN or −∞ likelihood is
                # mapped to the same penalty as an exception (audit RB-7):
                # every comparison Brent makes is False on NaN.
                # A trial τ beyond the family's reachable |τ| is evaluated at
                # the bound — the same density — and so is the result (F1).
                def _neg_ll(tau: float) -> float:
                    try:
                        ll = _eval_log_likelihood(
                            cls(tau_k=entry.reachable_tau(float(tau))), uv)
                    except Exception:
                        return MLE_FAIL_PENALTY
                    return -ll if np.isfinite(ll) else MLE_FAIL_PENALTY
                res   = minimize_scalar(_neg_ll, bounds=(tau_lo, tau_hi),
                                         method='bounded')
                tau_k = entry.reachable_tau(float(res.x))
                extra_vals = {}
                if not res.fun < MLE_FAIL_PENALTY:
                    converged = False
                    logger.warning(
                        "%s.fit(method='mle'): the likelihood is not finite at "
                        "any τ the search evaluated; returning τ = %.6g.",
                        class_name, tau_k,
                    )
            else:
                # Joint optimisation — the same routine as the ICE M-step.
                tau_hat, _ = _kendalltau(data[:, 0], data[:, 1])
                pfit = _fit_two_parameter_mle(cls, entry, uv, None, float(tau_hat))
                tau_k = float(pfit.params["tau_k"])
                extra_vals = {k: v for k, v in pfit.params.items() if k != "tau_k"}
                converged = pfit.converged
                if not converged:
                    logger.warning(
                        "%s.fit(method='mle'): joint optimisation did not "
                        "converge (%s); returning the best point found %s.",
                        class_name, pfit.message, pfit.params,
                    )

        else:
            raise ValueError(f"method must be 'tau' or 'mle', got {method!r}.")

        copula = cls(tau_k=tau_k, **extra_vals)
        return FitResult(copula=copula, method=method, tau_k=tau_k,
                         log_likelihood=_eval_log_likelihood(copula, uv),
                         n_obs=n, uv=uv, converged=converged)

    @staticmethod
    def fit_best(data: np.ndarray, families: list | None = None,
                 method: str = 'tau') -> 'FitBestResults':
        """Fit each candidate family and rank the comparable fits by AIC.

        Parameters
        ----------
        data     : array-like, shape (n, 2)
        families : list of CopulaVirt subclasses, or None for every available
                   family except Product.
        method   : passed to each cls.fit()

        Returns
        -------
        FitBestResults — a ``list[FitResult]`` sorted by AIC ascending, holding
        only the fits obtained **with the requested method**. Two attributes
        report what the ranking leaves out (audit RB-10):

        * ``other_method`` — fits the family could only produce with another
          method. Student and BB1 always fit by maximum likelihood (τ alone
          does not identify their second parameter): under ``method='tau'``
          their log-likelihood is a maximum, the others' the likelihood at the
          τ-inversion estimate — lower by construction — so ranking them
          together favoured the two-parameter families. ``method='mle'``
          ranks every family on maximised likelihoods.
        * ``failures`` — ``(class name, "ExceptionType: message")`` for every
          family whose fit raised (formerly logged and dropped).

        Both are also logged at WARNING. ``converged`` on each result says
        whether its likelihood optimiser converged.
        """
        if families is None:
            import importlib as _il
            families = []
            for _entry in CopulaEnum:
                if not _entry.value.AVAILABLE or not _entry.value.MODULE:
                    continue
                if _entry.value.CLASS_NAME == 'CopulaProduct':   # τ fixé à 0, pas de fit utile
                    continue
                _mod = _il.import_module(_entry.value.MODULE)
                families.append(getattr(_mod, _entry.value.CLASS_NAME))

        ranked: list = []
        other: list = []
        failures: list[tuple[str, str]] = []
        for cls in families:
            try:
                r = cls.fit(data, method=method)
            except Exception as e:
                logger.warning('%s.fit failed: %s', cls.__name__, e)
                failures.append((cls.__name__, f"{type(e).__name__}: {e}"))
                continue
            if r.method != method:
                logger.warning(
                    "fit_best(method=%r): %s was fitted by %r — its "
                    "log-likelihood is not comparable; reported in "
                    ".other_method, not ranked.", method, cls.__name__, r.method,
                )
                other.append(r)
            else:
                ranked.append(r)
        ranked.sort(key=lambda r: r.aic)
        other.sort(key=lambda r: r.aic)
        return FitBestResults(ranked, method=method, other_method=other,
                              failures=failures)

    def standard_errors(self, uv: np.ndarray, weights: np.ndarray | None = None,
                        method: str = 'mle', *, ranks: bool = True):
        """Asymptotic standard errors of this copula's parameters as estimated on ``uv``.

        Opt-in and side-effect free: nothing in :meth:`fit` changes. The
        copula's current parameters are taken as the estimate; ``uv`` are the
        pseudo-observations it was computed from, ``weights`` optional
        frequency weights (``n_eff = Σw``).

        * ``method='mle'`` — sandwich of the pseudo-likelihood with the
          estimated-rank corrections (Genest, Ghoudi & Rivest 1995,
          doi:10.1093/biomet/82.3.543); ``ranks=False`` for known margins.
        * ``method='tau'`` — 16·Var{2C(U,V) − U − V}/n for Kendall's τ̂, carried
          to θ by dθ/dτ (Genest & Favre 2007,
          doi:10.1061/(ASCE)1084-0699(2007)12:4(347); Kojadinovic & Yan 2010,
          doi:10.1016/j.insmatheco.2010.03.008). One-parameter families only.

        Returns a :class:`pmcprg.copulas._stderr.StandardErrors` (estimate, se,
        cov, ci(level), at_boundary, n_eff). At a boundary of the parameter
        space (independence end of Clayton/GH/Joe, BB1 δ = 1, Student ν at its
        bound) ``at_boundary`` is set and no Wald interval is reported (Self &
        Liang 1987, doi:10.1080/01621459.1987.10478472). See
        :mod:`pmcprg.copulas._stderr` for the formulas and their limits
        (i.i.d. pairs; FR-4 inside ICE and FR-5 not covered).
        """
        from pmcprg.copulas._stderr import standard_errors as _standard_errors
        return _standard_errors(self, uv, weights, method, ranks=ranks)

    # ------------------------------------------------------------------
    # Representations
    # ------------------------------------------------------------------
    def __repr__(self):
        return str(self)

    def __str__(self):
        return (
            f"Copula name = {self.copula_enum}; Tau Kendall= {self.params['tau_k']:.2f}"
        )

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------
    @with_package_style
    def plot_pdf(self, plot_dir: str, prefix: str = "") -> None:
        """Contour plot of the copula density c(u,v) on (0,1)²."""
        Z = np.vectorize(lambda a, b: self.pdf([a, b]))(self._x2, self._y2)
        fig, ax = plt.subplots(figsize=(5, 5))
        vmax = np.nanpercentile(Z, 97)
        vticks = np.linspace(0, max(vmax, 1e-10), self.ticks_nbr)
        # extend="max": the levels stop at the 97th percentile, so without it
        # contourf leaves every over-range cell UNFILLED — copula densities
        # diverge at the corners, so the regions of highest density (a tail
        # dependence, the very feature one looks for) rendered as white holes.
        cs = ax.contourf(
            self._x2, self._y2, Z, vticks, cmap="viridis",
            vmin=0, vmax=vmax, extend="max",
        )
        cbar = fig.colorbar(cs, ax=ax, ticks=vticks[::3])
        cbar.set_label("density c(u, v)")
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_aspect("equal")
        theta_str = f", θ={self.theta:.3f}" if hasattr(self, "theta") else ""
        fig.suptitle(
            f"{self.copula_enum.LONG_NAME}  c(u,v)  τ={self.params['tau_k']}{theta_str}",
            y=0.98,
        )
        plt.savefig(
            os.path.join(
                plot_dir,
                f"{prefix}PDF_{self.copula_enum.SHORT_NAME}_tau{self.params['tau_k']:.2f}.png",
            ),
            bbox_inches="tight",
        )
        plt.close()

    @with_package_style
    def plot_cdf(self, plot_dir: str, prefix: str = "") -> None:
        """Contour plot of the copula CDF C(u,v) on (0,1)²."""
        try:
            Z = np.vectorize(lambda a, b: self.cdf([a, b]))(self._x2, self._y2)
        except NotImplementedError as e:
            logger.warning("plot_cdf skipped: %s", e)
            return
        fig, ax = plt.subplots(figsize=(5, 5))
        vticks = np.linspace(0, 1, self.ticks_nbr)
        cs = ax.contourf(self._x2, self._y2, Z, vticks, cmap="viridis", vmin=0, vmax=1)
        fig.colorbar(cs, ax=ax, ticks=vticks[::3])
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_aspect("equal")
        theta_str = f", θ={self.theta:.3f}" if hasattr(self, "theta") else ""
        fig.suptitle(
            f"{self.copula_enum.LONG_NAME}  C(u,v)  τ={self.params['tau_k']}{theta_str}",
            y=0.98,
        )
        plt.savefig(
            os.path.join(
                plot_dir,
                f"{prefix}CDF_{self.copula_enum.SHORT_NAME}_tau{self.params['tau_k']:.2f}.png",
            ),
            bbox_inches="tight",
        )
        plt.close()

    # ------------------------------------------------------------------
    # Sampling via Rosenblatt / h-function inversion
    # ------------------------------------------------------------------

    def sample(self, n: int = 500, seed: int | None = None) -> np.ndarray:
        """Draw n samples from the copula on [0,1]² via Rosenblatt inversion.

        For each sample::

            u ~ Uniform(EPS, 1-EPS)
            w ~ Uniform(EPS, 1-EPS)
            v = inv_h(w, u)

        Subclasses with a closed-form ``inv_h_array`` (Gaussian, Clayton,
        Frank) get a single vectorised call for all n samples; others
        fall back to per-sample Brent inversion via ``inv_h``.
        """
        rng = np.random.default_rng(seed)
        us  = rng.uniform(EPS, ONE_MINUS_EPS, n)
        ws  = rng.uniform(EPS, ONE_MINUS_EPS, n)
        out = np.empty((n, 2))
        out[:, 0] = us
        out[:, 1] = self.inv_h_array(ws, us)
        return out

    # ------------------------------------------------------------------
    # New plots
    # ------------------------------------------------------------------

    @with_package_style
    def plot_samples(
        self, plot_dir: str, prefix: str = "", n: int = 5000, seed: int = 42
    ) -> None:
        """Scatter of n copula samples overlaid on the PDF background."""
        samples = self.sample(n, seed=seed)
        Z = np.vectorize(lambda a, b: self.pdf([a, b]))(self._x2, self._y2)
        vmax = np.nanpercentile(Z, 97)

        fig, ax = plt.subplots(figsize=(5, 5))
        ticks = np.linspace(0, max(vmax, 1e-10), 8)
        ax.contourf(
            self._x2, self._y2, Z, ticks, cmap="Blues", alpha=0.45,
            vmin=0, vmax=vmax, extend="max"
        )
        ax.scatter(
            samples[:, 0], samples[:, 1], s=4, alpha=0.5, color="navy", linewidths=0
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_aspect("equal")
        theta_str = f", θ={self.theta:.3f}" if hasattr(self, "theta") else ""
        fig.suptitle(
            f"{self.copula_enum.LONG_NAME}  τ={self.params['tau_k']}{theta_str}  (n={n})",
            y=0.98,
        )
        plt.savefig(
            os.path.join(
                plot_dir,
                f"{prefix}Samples_{self.copula_enum.SHORT_NAME}_tau{self.params['tau_k']:.2f}.png",
            ),
            bbox_inches="tight",
        )
        plt.close()

    @with_package_style
    def plot_h_function(self, plot_dir: str, prefix: str = "") -> None:
        """Heatmap of h(v|u) = ∂C(u,v)/∂u with iso-probability contours.

        For the independence copula h(v|u) = v → horizontal stripes.
        Iso-contours (dashed) show levels [0.1, 0.25, 0.5, 0.75, 0.9].
        """
        H = np.vectorize(lambda a, b: self.conditional_cdf(float(b), float(a)))(
            self._x2, self._y2
        )

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.pcolormesh(
            self._x2, self._y2, H, cmap="viridis", vmin=0, vmax=1, shading="auto"
        )
        fig.colorbar(im, ax=ax, label="h(v|u)")
        ax.contour(
            self._x2,
            self._y2,
            H,
            levels=[0.1, 0.25, 0.5, 0.75, 0.9],
            colors="white",
            linewidths=0.6,
            alpha=0.7,
            linestyles="--",
        )
        ax.set_xlabel("u  (conditioning)")
        ax.set_ylabel("v")
        ax.set_aspect("equal")
        theta_str = f", θ={self.theta:.3f}" if hasattr(self, "theta") else ""
        fig.suptitle(
            f"h(v|u) = ∂C/∂u  —  {self.copula_enum.LONG_NAME}  τ={self.params['tau_k']}{theta_str}",
            y=0.98,
        )
        plt.savefig(
            os.path.join(
                plot_dir,
                f"{prefix}Hfunc_{self.copula_enum.SHORT_NAME}_tau{self.params['tau_k']:.2f}.png",
            ),
            bbox_inches="tight",
        )
        plt.close()

    @with_package_style
    def plot_overview(
        self, plot_dir: str, prefix: str = "", n_samples: int = 2000, seed: int = 42
    ) -> None:
        """2×2 panel: PDF · CDF · h-function · samples."""
        tau = self.params["tau_k"]
        name = self.copula_enum.value.LONG_NAME
        theta_str = f", θ={self.theta:.3f}" if hasattr(self, "theta") else ""

        Z_pdf = np.vectorize(lambda a, b: self.pdf([a, b]))(self._x2, self._y2)
        Z_h = np.vectorize(lambda a, b: self.conditional_cdf(float(b), float(a)))(
            self._x2, self._y2
        )
        try:
            Z_cdf = np.vectorize(lambda a, b: self.cdf([a, b]))(self._x2, self._y2)
            has_cdf = True
        except NotImplementedError:
            has_cdf = False
        samples = self.sample(n_samples, seed=seed)

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle(f"{name}  (τ={tau}{theta_str})", fontsize=FONT_SIZE + 2)
        fig.subplots_adjust(top=0.93, wspace=0.32, hspace=0.32)

        # PDF
        ax = axes[0, 0]
        vmax = np.nanpercentile(Z_pdf, 97)
        tks = np.linspace(0, max(vmax, 1e-10), self.ticks_nbr)
        cs = ax.contourf(
            self._x2, self._y2, Z_pdf, tks, cmap="viridis",
            vmin=0, vmax=vmax, extend="max"
        )
        fig.colorbar(cs, ax=ax, ticks=tks[::3])
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_aspect("equal")
        ax.set_title("PDF  c(u, v)")

        # CDF
        ax = axes[0, 1]
        if has_cdf:
            tks = np.linspace(0, 1, self.ticks_nbr)
            cs = ax.contourf(
                self._x2, self._y2, Z_cdf, tks, cmap="viridis", vmin=0, vmax=1
            )
            fig.colorbar(cs, ax=ax, ticks=tks[::3])
            ax.set_xlabel("u")
            ax.set_ylabel("v")
            ax.set_aspect("equal")
        else:
            ax.text(
                0.5,
                0.5,
                "CDF not available\nin closed form",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        ax.set_title("CDF  C(u, v)")

        # h-function
        ax = axes[1, 0]
        im = ax.pcolormesh(
            self._x2, self._y2, Z_h, cmap="viridis", vmin=0, vmax=1, shading="auto"
        )
        fig.colorbar(im, ax=ax)
        ax.contour(
            self._x2,
            self._y2,
            Z_h,
            levels=[0.1, 0.25, 0.5, 0.75, 0.9],
            colors="white",
            linewidths=0.5,
            alpha=0.6,
            linestyles="--",
        )
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_aspect("equal")
        ax.set_title("h(v|u) = ∂C/∂u")

        # Samples
        ax = axes[1, 1]
        tks = np.linspace(0, max(vmax, 1e-10), 8)
        ax.contourf(
            self._x2, self._y2, Z_pdf, tks, cmap="Blues", alpha=0.4,
            vmin=0, vmax=vmax, extend="max",
        )
        ax.scatter(
            samples[:, 0], samples[:, 1], s=3, alpha=0.5, color="navy", linewidths=0
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_aspect("equal")
        ax.set_title(f"Samples  (n={n_samples})")

        plt.savefig(
            os.path.join(
                plot_dir,
                f"{prefix}Overview_{self.copula_enum.SHORT_NAME}_tau{tau:.2f}.png",
            ),
            bbox_inches="tight",
        )
        plt.close()

    @with_package_style
    def plot_multi_tau(
        self,
        plot_dir: str,
        prefix: str = "",
        tau_values: list | None = None,
        ncols: int = 3,
    ) -> None:
        """Grid of PDF contours for several τ values — illustrates the copula family."""
        tau_min, tau_max = self.tau_range
        if abs(tau_max - tau_min) < 1e-8:
            return
        if tau_values is None:
            span = tau_max - tau_min
            pad = min(0.1 * span, 0.05)
            tau_values = list(np.linspace(tau_min + pad, tau_max - pad, ncols * 2))

        n = len(tau_values)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4 * ncols, 4 * nrows),
            squeeze=False,
        )
        fig.suptitle(
            f"{self.copula_enum.value.LONG_NAME}  —  PDF c(u,v) for various τ",
            fontsize=FONT_SIZE + 2,
            y=1.01,
        )
        for idx, tau in enumerate(tau_values):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]
            cop = self.__class__(**{**self.params, 'tau_k': float(tau)})
            Z = np.vectorize(lambda a, b: cop.pdf([a, b]))(cop._x2, cop._y2)
            vmax = np.nanpercentile(Z, 97)
            tks = np.linspace(0, max(vmax, 1e-10), 9)
            ax.contourf(cop._x2, cop._y2, Z, tks, cmap="viridis",
                        vmin=0, vmax=vmax, extend="max")
            theta_str = f", θ={cop.theta:.2f}" if hasattr(cop, "theta") else ""
            ax.set_title(f"τ = {tau:.2f}{theta_str}")
            ax.set_xlabel("u")
            ax.set_ylabel("v")
            ax.set_aspect("equal")

        for idx in range(n, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                plot_dir, f"{prefix}MultiTau_{self.copula_enum.SHORT_NAME}.png"
            ),
            bbox_inches="tight",
        )
        plt.close()




# ---------------------------------------------------------------------------
# Fitting helpers — re-exported from pmcprg.copulas._fit for backward compatibility
# (consumers — bivariate.py, the public ``__init__``, and external code —
# import these symbols from ``_base``; the underscore-prefixed helpers are
# also used by the test suite).
# ---------------------------------------------------------------------------

from pmcprg.copulas._fit import (   # noqa: E402, F401  (re-export at module bottom)
    MLE_FAIL_PENALTY,
    _cvm_statistic,
    _empirical_copula,
    _empirical_tail_dep,
    _eval_log_likelihood,
    _fit_two_parameter_mle,
    _weighted_log_density_sum,
    FitBestResults,
    FitResult,
    GoFResult,
    ParameterFit,
)
