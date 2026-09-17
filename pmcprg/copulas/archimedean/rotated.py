"""
90°/270°-rotation copulas (audit FR-8: Clayton, GH, Joe, BB1, A12, A14, and
BB1's own 180°/survival rotation — ``SurvivalBB190``/``SurvivalBB1270``,
closing round, section near the bottom of this file).

Given a base copula C with density c and h-function h(v|u) = ∂C/∂u:

    90°  (clockwise):          C90(u,v)  = v − C(1−u, v)
    270° (counter-clockwise):  C270(u,v) = u − C(u, 1−v)

Both give negative Kendall's τ (τ_rot = −τ_base) but are *not* the same
copula: 90° reflects only the first argument, 270° only the second, so a
one-sided base family (Clayton: lower-tail dependence at (u,v)→(0,0) only)
rotates into two distinct negative-dependence copulas with opposite tail
asymmetry. (U,V)~C with mass concentrated near (0,0): 90° maps it to
(1−U, V), pushing that mass to (u,v)→(1,0), the lower-right corner; 270° maps
it to (U, 1−V), pushing it to (u,v)→(0,1), the upper-left corner. Verified
numerically in ``pmcprg/tests/test_rotated_clayton.py`` by simulating both
and comparing corner masses (≈2.7× more mass in the lower-right corner than
the upper-left one for Clayton90 at τ = −0.6, and the mirror image for
Clayton270).

Derivation (also Brechmann & Schepsmeier 2013, "Modeling Dependence with C-
and D-Vine Copulas: The R Package CDVine", §2.1; Joe 2014, *Dependence
Modeling with Copulas*, §1.6): if (U, V) ~ C, then (1−U, V) has the 90°
copula as its law (U' = 1−U is a decreasing transform of U, so nothing about
V's margin or the (U,V) coupling other than the sign of τ changes), and
(U, 1−V) has the 270° copula as its law. Reading the CDF, density and
h-function off that change of variables — carefully, term by term, since
only *one* derivative is taken through the reflected argument for the CDF
sign and only *one* argument is reflected in the h-function's "1 −" term:

    density         : c90(u,v)  = c(1−u, v)              [no sign flip — the
                        two sign flips of d(1−u)/du in C90 = v − C(1−u,v)
                        cancel across the mixed partial ∂²/∂u∂v]
                       c270(u,v) = c(u, 1−v)              [symmetric argument]

    h-function       : h90(v|u)  = ∂C90/∂u  = h_C(v | 1−u)        — the
                        conditioning variable is reflected, v is not, and
                        there is *no* "1 −" (a single sign flip from
                        d(1−u)/du cancels the sign flip already inside
                        ∂/∂u[−C(1−u,v)])
                       h270(v|u) = ∂C270/∂u = 1 − h_C(1−v | u)    — here v
                        (not the conditioning variable) is reflected, and the
                        "1 −" is genuine: ∂C270/∂u = u − C(u,1−v) differentiates
                        the un-reflected u term to 1, so the sign is not
                        cancelled the way it is for 90°.

    inverse h        : inv_h90(w,u)  = invh_C(w | 1−u)
                       inv_h270(w,u) = 1 − invh_C(1−w | u)

    Kendall's τ       : τ90 = τ270 = −τ_C  (a decreasing transform of one
                        margin flips the sign of concordance; Nelsen 2006,
                        Theorem 5.1.3)

This file provides a generic wrapper, ``RotatedCopula90``/``RotatedCopula270``,
that composes any base family already exposing the *kernel interface* used by
the 180° (survival) rotation in ``survival.py`` — ``_kcoord``,
``_kcoord_reflected``, ``_k_logpdf``, ``_k_cdf``, ``_k_h``, ``_k_inv_h`` — with
the reflection above, so the two matching evaluations at (1−u, v) or (u, 1−v)
never form the complement in floating point (same numerical motivation as
``survival.py``'s module docstring, but only one coordinate is reflected here,
so the loss of precision near u = 0 or v = 1 is half as severe as the 180°
case's simultaneous (1−u, 1−v)).

Concrete subclasses: ``CopulaClayton90``/``CopulaClayton270`` (FR-8 pilot);
``CopulaGH90``/``CopulaGH270``/``CopulaJoe90``/``CopulaJoe270`` (FR-8, second
round) — both GH and Joe already exposed the kernel interface (used by
``SurvivalGH``/``SurvivalJoe``), so each pair is the same two-line subclass
as Clayton's, exactly as the pilot's module docstring predicted; and
``CopulaBB190``/``CopulaBB1270`` (FR-8, third round) — BB1 did *not* expose
the kernel interface, so ``pmcprg.copulas.archimedean.bb1`` was refactored
first (its ``cdf``/``pdf``/``conditional_cdf`` are now built from
``_kcoord``/``_k_logpdf``/``_k_cdf``/``_k_h``, exactly like GH/Joe, with the
existing tests passing bit-identically) before the two-line subclass pattern
below could apply to it too. BB1's own kernel coordinate turned out to
coincide with GH's (``ka = log u``, not Joe's ``log(1 − u)``), since BB1's
generator ``(t^{-θ} − 1)^δ`` is built from ``t`` directly; but BB1's
``_k_inv_h`` has no closed form (module docstring of ``bb1.py``) and solves
``h(v|u) = w`` by Brent's method on the kernel coordinate itself rather than
a closed-form Newton step, which is still a genuine kernel-interface
implementation, not a compromise on the mechanism here. Because BB1 carries
a second parameter (``delta``), it also needed :meth:`RotatedCopula.
constrain_params`/:meth:`RotatedCopula.constructible_params` overrides
(below) that Clayton/GH/Joe's rotations never needed (their base classes'
versions are the identity).

``CopulaA1290``/``CopulaA12270``/``CopulaA1490``/``CopulaA14270`` (FR-8,
last round) close FR-8 for all six one-signed families. A12 and A14 did not
expose the kernel interface either, and were refactored the same way as BB1
(public values unchanged bit for bit): A14's kernel coordinate is ``log u``
(GH/BB1's), while A12's is the *pair* ``(log u, log(1 − u))``, since its
generator ``(1/t − 1)^θ`` needs both logarithms separately — this module
never looks inside a kernel coordinate, so a tuple is as good as a float
here (module docstring of ``a12.py``). Both families' h-functions share
the shape ``(a/t)^{θ−1}((1 + a)/(1 + t))^p`` in their transformed
coordinate ``a``, so their ``_k_inv_h`` is one monotone Newton iteration
with closed-form bounds (vectorised, ulp-accurate), not BB1's Brent loop.
Their registered range ``[−1, −1/3]`` is the mirror of ``[1/3, 1]``: the
first rotated range whose inner end is *not* independence, which the
range-generic machinery (``constructible_tau_range``, ``correct_tau``,
multistart family draws, the Huard grid, the selection placeholder)
handles unchanged — see ``test_rotated_a12_a14.py``; the one range check
that did not generalise was ``independence_lr_test``, which recognised
independence only at a *lower* range end and so refused the older
rotations (range ``[−1, −ε]``) as if they lacked it, like A12. (In the
tests, ``test_scientific.py``'s generic sweep clipped τ = +0.5 into each
range, so it had only ever checked the earlier rotations at τ = −ε; it now
sweeps negative ranges at −0.5.) A12 and A14 have
tail dependence in both diagonal corners (A12: λ_L = 2^{−1/θ},
λ_U = 2 − 2^{1/θ}; A14: λ_L = 1/2, λ_U = 2 − 2^{1/θ}), so, like BB1's,
their rotations populate both anti-diagonal corners; for A14 the dominant
one changes with τ (section comment below).

Reference: Brechmann, E. C. & Schepsmeier, U. (2013). *Journal of Statistical
Software* 52(3); Joe, H. (2014). *Dependence Modeling with Copulas*, Chapman
& Hall/CRC, §1.6 (rotated/reflected copulas, τ_rot = −τ).
"""
if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import logging
import numpy as np

from pmcprg.copulas._base               import CopulaVirt
from pmcprg.copulas._fit                import FitResult
from pmcprg.copulas.archimedean.a12     import CopulaA12
from pmcprg.copulas.archimedean.a14     import CopulaA14
from pmcprg.copulas.archimedean.bb1     import CopulaBB1
from pmcprg.copulas.archimedean.clayton import CopulaClayton
from pmcprg.copulas.archimedean.gumbel  import CopulaGH
from pmcprg.copulas.archimedean.joe     import CopulaJoe
from pmcprg.copulas.archimedean.survival import SurvivalBB1
from pmcprg.numerics                 import EPS, ONE_MINUS_EPS, minmaxEPS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generic wrapper — parameter plumbing shared by 90° and 270°
# ---------------------------------------------------------------------------

class RotatedCopula(CopulaVirt):
    """Common machinery for a 90°/270°-rotation wrapper around a base copula.

    Subclasses set the class attribute ``_base_class`` to a family that
    implements the kernel interface (module docstring) and mix in either
    :class:`RotatedCopula90` or :class:`RotatedCopula270` for the per-angle
    formulas — see the concrete ``CopulaClayton90``/``CopulaClayton270``
    below for the two-line pattern a sixth family would add.

    Every parameter other than ``tau_k`` (e.g. BB1's ``delta``, a fast
    follow, not this round) is passed through to the base copula unchanged —
    a rotation reflects the copula's support, not its shape parameters — and
    mirrored back onto ``self`` for the plotting helpers (``theta``,
    ``delta``) that read it off the instance.
    """

    _base_class: type[CopulaVirt] | None = None

    _KERNEL_INTERFACE = ('_kcoord', '_kcoord_reflected', '_k_logpdf', '_k_cdf', '_k_h', '_k_inv_h')

    def __init__(self, **kwargs):
        missing = [m for m in self._KERNEL_INTERFACE if not hasattr(self._base_class, m)]
        if missing:
            raise TypeError(f"{type(self).__name__}: base {getattr(self._base_class, '__name__', None)} "
                            f"lacks the kernel interface {missing} (see pmcprg.copulas.archimedean.rotated).")
        # Build the base copula first; _update_params (called by super) syncs it.
        base_kwargs = dict(kwargs)
        base_kwargs['tau_k'] = -kwargs['tau_k']
        self._base: CopulaVirt = self._base_class(**base_kwargs)  # type: ignore[misc]
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    def _update_params(self):
        """Propagate current params to the base copula (τ negated, everything
        else unchanged) and mirror its derived attributes."""
        base_params = dict(self.params)
        base_params['tau_k'] = -self.params['tau_k']
        self._base.params = base_params
        self._base._update_params()
        if hasattr(self._base, 'theta'):
            self.theta = self._base.theta
        if hasattr(self._base, 'delta'):
            self.delta = self._base.delta

    @staticmethod
    def _columns(uv):
        uv = np.asarray(uv, dtype=float)
        return (np.clip(uv[:, 0], EPS, ONE_MINUS_EPS),
                np.clip(uv[:, 1], EPS, ONE_MINUS_EPS))

    # ------------------------------------------------------------------
    # Joint-constraint delegation (audit G1, needed once a base family has
    # more than τ alone — BB1's ``delta``). ``tau_k`` is negated both ways
    # around the base family's own hook, since the rotated family's public τ
    # is minus the base's; for Clayton/GH/Joe, whose base ``constrain_params``/
    # ``constructible_params`` are :class:`CopulaVirt`'s identity, this
    # round-trips to the same dict (negation is exact), so nothing changes
    # for the existing one-parameter rotations.
    # ------------------------------------------------------------------
    @classmethod
    def constrain_params(cls, params: dict) -> dict:
        base_params = dict(params)
        base_params['tau_k'] = -base_params['tau_k']
        base_params = cls._base_class.constrain_params(base_params)
        base_params['tau_k'] = -base_params['tau_k']
        return base_params

    @classmethod
    def constructible_params(cls, params: dict) -> dict:
        """``params`` itself, unchanged, when the base's own hook would not
        touch it either — :meth:`CopulaVirt.constructible_params`'s identity
        contract (``params`` object returned as-is, no copy) matters to
        callers that compare by identity (e.g. the multistart machinery's
        own "was this pair repaired?" check), so it is preserved through the
        τ-negation round-trip: only when the base's repaired dict actually
        differs from what was handed to it is a new (negated) dict built and
        returned."""
        base_params = dict(params)
        base_params['tau_k'] = -base_params['tau_k']
        repaired = cls._base_class.constructible_params(base_params)
        if repaired is base_params:
            return params
        repaired = dict(repaired)
        repaired['tau_k'] = -repaired['tau_k']
        return repaired

    # ------------------------------------------------------------------
    # Per-angle formulas — overridden by RotatedCopula90 / RotatedCopula270
    # ------------------------------------------------------------------
    def _logpdf(self, u, v):
        raise NotImplementedError

    def _cdf(self, u, v):
        raise NotImplementedError

    def _h(self, v, u):
        raise NotImplementedError

    def _inv_h(self, w, u):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public API — identical shape to survival.py / every other family
    # ------------------------------------------------------------------
    def pdf(self, uv):
        u = np.float64(minmaxEPS(uv[0]))
        v = np.float64(minmaxEPS(uv[1]))
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            return float(np.exp(self._logpdf(u, v)))

    def pdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised density, ``exp`` of :meth:`logpdf_array` (no floor)."""
        return np.exp(self.logpdf_array(uv))

    def logpdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised log-density; the reflected argument is never formed as
        a floating-point complement (module docstring)."""
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            return self._logpdf(*self._columns(uv))

    def cdf(self, uv):
        u = np.float64(minmaxEPS(uv[0]))
        v = np.float64(minmaxEPS(uv[1]))
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            return float(self._cdf(u, v))

    def cdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised rotated CDF (module docstring: ``v − C(1−u,v)`` or
        ``u − C(u,1−v)``)."""
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            return self._cdf(*self._columns(uv))

    def conditional_cdf(self, v: float, u: float) -> float:
        """h_rot(v|u), the kernel member selected by the angle mixin."""
        u = np.float64(minmaxEPS(u))
        v = np.float64(minmaxEPS(v))
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            h = self._h(v, u)
        return float(np.clip(h, 0.0, 1.0))

    def inv_h(self, w: float, u: float) -> float:
        return float(self.inv_h_array(np.array([w], dtype=float), np.array([u], dtype=float))[0])

    def inv_h_array(self, w: np.ndarray, u: np.ndarray) -> np.ndarray:
        w = np.clip(np.asarray(w, dtype=float), EPS, ONE_MINUS_EPS)
        u = np.clip(np.asarray(u, dtype=float), EPS, ONE_MINUS_EPS)
        if w.shape != u.shape:
            raise ValueError(f"inv_h_array: w {w.shape} and u {u.shape} must match.")
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            v = self._inv_h(w, u)
        return np.clip(v, EPS, ONE_MINUS_EPS)

    def tail_dependence(self) -> tuple[float, float]:
        """(0, 0) — a 90°/270° rotation of a one-signed family has no
        *diagonal* tail dependence in the usual (λ_L, λ_U) sense: negative
        dependence concentrates near the anti-diagonal, which
        :meth:`CopulaVirt.tail_dependence`'s ``C(u,u)``/``C(1−u,1−u)`` probe
        (built for positive/see-survival dependence) does not see. This
        matches VineCopula's own convention for its rotated one-parameter
        families (only the 180° survival rotation carries the base family's
        tail dependence, corners swapped)."""
        return 0.0, 0.0


class RotatedCopula90(RotatedCopula):
    """90°-rotation formulas: C90(u,v) = v − C(1−u, v) (module docstring)."""

    def _logpdf(self, u, v):
        b = self._base
        return b._k_logpdf(b._kcoord_reflected(u), b._kcoord(v))

    def _cdf(self, u, v):
        b = self._base
        c, _ = b._k_cdf(b._kcoord_reflected(u), b._kcoord(v))
        return np.clip(v - c, 0.0, 1.0)

    def _h(self, v, u):
        """h90(v|u) = h_C(v | 1−u) — no ``1 −`` (module docstring)."""
        b = self._base
        h, _ = b._k_h(b._kcoord(v), b._kcoord_reflected(u))
        return h

    def _inv_h(self, w, u):
        """inv_h90(w,u) = invh_C(w | 1−u)."""
        b = self._base
        v, _ = b._k_inv_h(np.log(w), b._kcoord_reflected(u))
        return v


class RotatedCopula270(RotatedCopula):
    """270°-rotation formulas: C270(u,v) = u − C(u, 1−v) (module docstring)."""

    def _logpdf(self, u, v):
        b = self._base
        return b._k_logpdf(b._kcoord(u), b._kcoord_reflected(v))

    def _cdf(self, u, v):
        b = self._base
        c, _ = b._k_cdf(b._kcoord(u), b._kcoord_reflected(v))
        return np.clip(u - c, 0.0, 1.0)

    def _h(self, v, u):
        """h270(v|u) = 1 − h_C(1−v | u) — genuine ``1 −`` (module docstring)."""
        b = self._base
        _, hbar = b._k_h(b._kcoord_reflected(v), b._kcoord(u))
        return hbar

    def _inv_h(self, w, u):
        """inv_h270(w,u) = 1 − invh_C(1−w | u)."""
        b = self._base
        _, vbar = b._k_inv_h(np.log1p(-w), b._kcoord(u))
        return vbar


# ---------------------------------------------------------------------------
# Concrete rotated copulas — Clayton pilot (FR-8)
# ---------------------------------------------------------------------------

class CopulaClayton90(RotatedCopula90):
    """90°-rotated Clayton copula — negative dependence, mass concentrated
    near the lower-right corner (u→1, v→0).

    τ = −τ_Clayton(θ), θ built from |τ| at the base Clayton. λ_L = λ_U = 0
    in the usual diagonal sense (see :meth:`RotatedCopula.tail_dependence`).
    """
    _base_class = CopulaClayton


class CopulaClayton270(RotatedCopula270):
    """270°-rotated Clayton copula — negative dependence, mass concentrated
    near the upper-left corner (u→0, v→1).

    Same τ(θ) map as :class:`CopulaClayton90` (both negate the base τ), but
    a *different* copula: the reflection acts on the other margin, so the
    two families put their residual asymmetric mass on opposite corners
    (numerically verified in ``pmcprg/tests/test_rotated_clayton.py``).
    """
    _base_class = CopulaClayton


# ---------------------------------------------------------------------------
# Concrete rotated copulas — GH and Joe (FR-8, second round)
# ---------------------------------------------------------------------------
#
# Both already expose the kernel interface (used by SurvivalGH/SurvivalJoe
# in survival.py), so — exactly as the pilot's own module docstring
# predicted — each is the same two-line subclass as Clayton above.

class CopulaGH90(RotatedCopula90):
    """90°-rotated Gumbel-Hougaard copula — negative dependence, mass
    concentrated near the lower-right corner (u→1, v→0).

    τ = −τ_GH(θ), θ built from |τ| at the base GH. λ_L = λ_U = 0 in the
    usual diagonal sense (see :meth:`RotatedCopula.tail_dependence`).
    """
    _base_class = CopulaGH


class CopulaGH270(RotatedCopula270):
    """270°-rotated Gumbel-Hougaard copula — negative dependence, mass
    concentrated near the upper-left corner (u→0, v→1).

    Same τ(θ) map as :class:`CopulaGH90` (both negate the base τ), but a
    *different* copula: the reflection acts on the other margin, so the two
    families put their residual asymmetric mass on opposite corners.
    """
    _base_class = CopulaGH


class CopulaJoe90(RotatedCopula90):
    """90°-rotated Joe copula — negative dependence, mass concentrated near
    the lower-right corner (u→1, v→0).

    τ = −τ_Joe(θ), θ built from |τ| at the base Joe. λ_L = λ_U = 0 in the
    usual diagonal sense (see :meth:`RotatedCopula.tail_dependence`).
    """
    _base_class = CopulaJoe


class CopulaJoe270(RotatedCopula270):
    """270°-rotated Joe copula — negative dependence, mass concentrated near
    the upper-left corner (u→0, v→1).

    Same τ(θ) map as :class:`CopulaJoe90` (both negate the base τ), but a
    *different* copula: the reflection acts on the other margin, so the two
    families put their residual asymmetric mass on opposite corners.
    """
    _base_class = CopulaJoe


# ---------------------------------------------------------------------------
# Concrete rotated copulas — BB1 (FR-8, third round)
# ---------------------------------------------------------------------------
#
# BB1 needed its kernel interface added first (``bb1.py``, this round) before
# the same two-line subclass pattern above could apply. Unlike Clayton/GH/Joe,
# BB1 carries a second free parameter (``delta``), passed through unchanged by
# ``RotatedCopula._update_params``, and its ``constrain_params``/
# ``constructible_params`` (module docstring) are genuinely exercised here —
# BB1 is the first base family for which they are not the identity.
#
# BB1's own (unrotated) tail asymmetry is *two-sided* (λ_L, λ_U both > 0 in
# general — a Joe-Clayton hybrid, module docstring of ``bb1.py``), unlike
# Clayton (pure lower) or GH/Joe (pure upper): a plain BB1 sample therefore
# has mass concentrated near *both* (0,0) and (1,1), in a ratio set by
# (τ, δ) rather than fixed. The 90°/270° rotations still each reflect only
# one margin, so BB190 pushes the (0,0) mass to (1,0) and the (1,1) mass to
# (0,1) — both corners populated, mirrored across the anti-diagonal from
# BB1270 — rather than concentrating in a single corner the way Clayton90/270
# or GH90/270/Joe90/270 do. See ``test_rotated_bb1.py`` for the sampling
# check that measures this directly instead of assuming a single-corner
# signature.

class CopulaBB190(RotatedCopula90):
    """90°-rotated BB1 copula — negative dependence, with residual mass in
    *both* the lower-right (u→1, v→0) and upper-left (u→0, v→1) corners (the
    90° reflection of BB1's own two-sided tail asymmetry — see the section
    docstring above), not a single dominant corner as for the one-signed
    rotated families.
    """
    _base_class = CopulaBB1

    @classmethod
    def fit(cls, data: np.ndarray, method: str = 'mle') -> 'FitResult':
        """Delegate to :meth:`CopulaBB1.fit` on the ``(1 − u, v)``-reflected
        data — module docstring: ``(U,V) ~ BB190`` iff ``(1−U,V) ~ BB1`` — τ̂
        negated back and rewrapped as ``CopulaBB190``.

        This sidesteps ``pmcprg.copulas._fit._fit_two_parameter_mle``'s
        ``delta`` branch, whose ``τ = (δθ + 2(δ−1))/(δ(θ+2))`` parametrisation
        (and its θ-bound built from the padded τ *ceiling*) is BB1's own,
        hardcoded to a positive τ range; on BB190/BB1270's negative range it
        returned a τ̂ *outside the registered range* (found in practice while
        exercising this fit, exactly what this round set out to check) rather
        than raising, because the box it searches never matches the sign of
        the family it is fitting. Reusing BB1's own already-validated 2-D
        MLE via the reflection identity avoids that mismatch entirely instead
        of teaching the generic two-parameter fit about a rotation's sign.
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError(f'data must be shape (n, 2), got {data.shape}.')
        reflected = np.column_stack([1.0 - data[:, 0], data[:, 1]])
        base_fit = CopulaBB1.fit(reflected, method=method)
        tau_k = -base_fit.tau_k
        cop = cls(tau_k=tau_k, delta=base_fit.copula.delta)
        uv = np.column_stack([1.0 - base_fit.uv[:, 0], base_fit.uv[:, 1]])
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            log_lik = float(np.sum(cop.logpdf_array(uv)))
        return FitResult(copula=cop, method='mle', tau_k=tau_k,
                         log_likelihood=log_lik, n_obs=base_fit.n_obs, uv=uv,
                         converged=base_fit.converged)


class CopulaBB1270(RotatedCopula270):
    """270°-rotated BB1 copula — same τ(θ, δ) map as :class:`CopulaBB190`
    (both negate the base τ, δ unchanged), but a *different* copula: the
    reflection acts on the other margin, so the two corner masses are
    mirrored across the anti-diagonal relative to BB190's.
    """
    _base_class = CopulaBB1

    @classmethod
    def fit(cls, data: np.ndarray, method: str = 'mle') -> 'FitResult':
        """Delegate to :meth:`CopulaBB1.fit` on the ``(u, 1 − v)``-reflected
        data, mirroring :meth:`CopulaBB190.fit` (see its docstring for why)."""
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError(f'data must be shape (n, 2), got {data.shape}.')
        reflected = np.column_stack([data[:, 0], 1.0 - data[:, 1]])
        base_fit = CopulaBB1.fit(reflected, method=method)
        tau_k = -base_fit.tau_k
        cop = cls(tau_k=tau_k, delta=base_fit.copula.delta)
        uv = np.column_stack([base_fit.uv[:, 0], 1.0 - base_fit.uv[:, 1]])
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            log_lik = float(np.sum(cop.logpdf_array(uv)))
        return FitResult(copula=cop, method='mle', tau_k=tau_k,
                         log_likelihood=log_lik, n_obs=base_fit.n_obs, uv=uv,
                         converged=base_fit.converged)


# ---------------------------------------------------------------------------
# Concrete rotated copulas — A12 and A14 (FR-8, last round)
# ---------------------------------------------------------------------------
#
# Both needed the kernel interface added first (``a12.py``/``a14.py``, this
# round), then take the same two-line subclass as every family above. They
# are one-parameter families: the inherited ``constrain_params``/
# ``constructible_params`` delegation is the identity for them, and the
# generic ``CopulaVirt.fit`` (bounded search on the padded registered range
# ``[−1, −1/3]``) needs no override, unlike BB1's.
#
# Tail dependence of the bases is two-sided — A12: λ_L = 2^{−1/θ} ≥ 1/2,
# λ_U = 2 − 2^{1/θ}; A14: λ_L = 1/2 at every θ, λ_U = 2 − 2^{1/θ} — so the
# lower (0,0) mass goes to (1,0) under 90° and to (0,1) under 270°, and the
# upper (1,1) mass to (0,1) and (1,0) respectively. For A12,
# λ_L − λ_U = 2^{−1/θ} + 2^{1/θ} − 2 ≥ 0 at every θ, so A1290's lower-right
# corner never loses to its upper-left one: measured on 40 000 points (corner
# side 0.05), 0.0271 vs 0.0056 at τ = −0.35, 0.0379 vs 0.0323 at −0.7, near
# parity at −0.9 (both tails → 1). For A14, λ_L = 1/2 is crossed by λ_U at
# θ = 1/log₂1.5 ≈ 1.71 (τ ≈ 0.547), and the dominant corner flips with it:
# A1490 has 0.0267 vs 0.0061 (lower-right first) at τ = −0.35 but 0.0318 vs
# 0.0358 (upper-left first) at τ = −0.7. The 270° rotations mirror each
# case. ``test_rotated_a12_a14.py`` measures these instead of assuming them.

class CopulaA1290(RotatedCopula90):
    """90°-rotated A12 copula (Nelsen 4.2.12) — τ ∈ [−1, −1/3], mass in
    both anti-diagonal corners (section comment above)."""
    _base_class = CopulaA12


class CopulaA12270(RotatedCopula270):
    """270°-rotated A12 copula — same τ(θ) map as :class:`CopulaA1290`, the
    corner masses mirrored across the anti-diagonal."""
    _base_class = CopulaA12


class CopulaA1490(RotatedCopula90):
    """90°-rotated A14 copula (Nelsen 4.2.14) — τ ∈ [−1, −1/3], mass in
    both anti-diagonal corners (section comment above)."""
    _base_class = CopulaA14


class CopulaA14270(RotatedCopula270):
    """270°-rotated A14 copula — same τ(θ) map as :class:`CopulaA1490`, the
    corner masses mirrored across the anti-diagonal."""
    _base_class = CopulaA14


# ---------------------------------------------------------------------------
# Concrete rotated copulas — survival BB1 (FR-8, closing round)
# ---------------------------------------------------------------------------
#
# The audit's FR-8 list of families needing 90°/270° rotations names "BB1 de
# survie" (survival BB1) separately from plain BB1 — so, like every base
# family above, its own 90°/270° children are built here, over
# ``_base_class = SurvivalBB1`` (``pmcprg.copulas.archimedean.survival``)
# rather than ``CopulaBB1`` directly. ``SurvivalBB1`` already exposes the
# kernel interface (inherited unchanged from ``CopulaBB1`` through
# ``SurvivalCopula``'s ``_kcoord_reflected``/``_k_logpdf``/... delegation),
# so — like GH/Joe before it — the two-line subclass pattern applies with no
# further refactor, and the same ``constrain_params``/``constructible_params``
# joint-constraint delegation this module needed for plain ``CopulaBB190``
# (module docstring) is already in place one level down, in
# ``SurvivalCopula`` itself (``survival.py``), so nothing extra is needed
# here either.
#
# A worked-out identity, checked numerically in
# ``pmcprg/tests/test_survival_bb1.py`` rather than left as an algebraic
# claim: composing the 180° (survival) and 90°/270° rotations in this order
# lands back on plain BB1's own 90°/270° rotations, swapped —
#     SurvivalBB190(u,v)  = v − [(1−u)+v−1+BB1(u,1−v)] = u − BB1(u,1−v) = CopulaBB1270(u,v)
#     SurvivalBB1270(u,v) = u − [u+(1−v)−1+BB1(1−u,v)] = v − BB1(1−u,v) = CopulaBB190(u,v)
# (both at the *same* (τ, δ)) — an instance of the dihedral composition rule
# 180° ∘ 90° = 270° (and 180° ∘ 270° = 90°) for a copula group whose base is
# exchangeable, which BB1 is (``bb1.py`` module docstring: ``c(u,v) =
# c(v,u)``). This makes ``SurvivalBB190``/``SurvivalBB1270`` mathematically
# redundant with the already-registered ``CopulaBB1270``/``CopulaBB190`` —
# but the audit names "BB1 de survie" as its own family needing both
# rotations, so both are registered here as their own ``CopulaEnum`` entries
# (IDs 35-37 with ``SurvivalBB1`` itself) rather than left unimplemented or
# silently aliased, keeping FR-8's own family list complete and explicit.
#
# BB1's own tail asymmetry is two-sided and swapped by the 180° rotation
# (λ_L(Ŝ) = λ_U(BB1), λ_U(Ŝ) = λ_L(BB1), ``survival.py``'s ``SurvivalBB1``
# docstring); the 90°/270° reflection then pushes that swapped two-sided
# mass into the anti-diagonal corners exactly as plain BB190/BB1270's own
# section comment above describes for BB1's un-swapped asymmetry — the
# identity above says this is not a coincidence, it is the same corner
# geometry (``test_survival_bb1.py`` measures it directly rather than
# assuming the mirror of BB190/BB1270's own numbers applies unchanged).

class SurvivalBB190(RotatedCopula90):
    """90°-rotated survival BB1 copula (FR-8, closing round) — negative
    dependence, mass in both anti-diagonal corners, mathematically identical
    to :class:`CopulaBB1270` at the same (τ, δ) (section comment above)."""
    _base_class = SurvivalBB1

    @classmethod
    def fit(cls, data: np.ndarray, method: str = 'mle') -> 'FitResult':
        """Delegate to :meth:`SurvivalBB1.fit` on the ``(1 − u, v)``-reflected
        data, mirroring :meth:`CopulaBB190.fit` one level up (its docstring's
        reasoning applies unchanged: the generic two-parameter fit's BB1
        ``delta`` branch is sign-aware, but reusing the base family's own
        validated 2-D MLE via the reflection identity is simpler and more
        robust than exercising that branch's negative-range path directly)."""
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError(f'data must be shape (n, 2), got {data.shape}.')
        reflected = np.column_stack([1.0 - data[:, 0], data[:, 1]])
        base_fit = SurvivalBB1.fit(reflected, method=method)
        tau_k = -base_fit.tau_k
        cop = cls(tau_k=tau_k, delta=base_fit.copula.delta)
        uv = np.column_stack([1.0 - base_fit.uv[:, 0], base_fit.uv[:, 1]])
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            log_lik = float(np.sum(cop.logpdf_array(uv)))
        return FitResult(copula=cop, method='mle', tau_k=tau_k,
                         log_likelihood=log_lik, n_obs=base_fit.n_obs, uv=uv,
                         converged=base_fit.converged)


class SurvivalBB1270(RotatedCopula270):
    """270°-rotated survival BB1 copula — same τ(θ, δ) map as
    :class:`SurvivalBB190` (both negate the base τ, δ unchanged),
    mathematically identical to :class:`CopulaBB190` at the same (τ, δ)
    (section comment above)."""
    _base_class = SurvivalBB1

    @classmethod
    def fit(cls, data: np.ndarray, method: str = 'mle') -> 'FitResult':
        """Delegate to :meth:`SurvivalBB1.fit` on the ``(u, 1 − v)``-reflected
        data, mirroring :meth:`SurvivalBB190.fit` (see its docstring for
        why)."""
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError(f'data must be shape (n, 2), got {data.shape}.')
        reflected = np.column_stack([data[:, 0], 1.0 - data[:, 1]])
        base_fit = SurvivalBB1.fit(reflected, method=method)
        tau_k = -base_fit.tau_k
        cop = cls(tau_k=tau_k, delta=base_fit.copula.delta)
        uv = np.column_stack([base_fit.uv[:, 0], 1.0 - base_fit.uv[:, 1]])
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            log_lik = float(np.sum(cop.logpdf_array(uv)))
        return FitResult(copula=cop, method='mle', tau_k=tau_k,
                         log_likelihood=log_lik, n_obs=base_fit.n_obs, uv=uv,
                         converged=base_fit.converged)


if __name__ == '__main__':
    from pathlib import Path

    for cls, name in [
        (CopulaClayton90,  'CopulaClayton90'),
        (CopulaClayton270, 'CopulaClayton270'),
        (CopulaGH90,       'CopulaGH90'),
        (CopulaGH270,      'CopulaGH270'),
        (CopulaJoe90,      'CopulaJoe90'),
        (CopulaJoe270,     'CopulaJoe270'),
        (CopulaBB190,      'CopulaBB190'),
        (CopulaBB1270,     'CopulaBB1270'),
        (CopulaA1290,      'CopulaA1290'),
        (CopulaA12270,     'CopulaA12270'),
        (CopulaA1490,      'CopulaA1490'),
        (CopulaA14270,     'CopulaA14270'),
        (SurvivalBB190,    'SurvivalBB190'),
        (SurvivalBB1270,   'SurvivalBB1270'),
    ]:
        cop = cls(tau_k=-0.5)
        print(f'\n--- {name} ---')
        print(f'tau_k    : {cop.params["tau_k"]:.4f}  range={cop.tau_range}')
        print(f'theta    : {cop.theta:.6f}')
        print(f'pdf(0.3, 0.7) = {cop.pdf([0.3, 0.7]):.6f}')
        print(f'cdf(0.3, 0.7) = {cop.cdf([0.3, 0.7]):.6f}')
        print(f'h(0.7 | 0.3)  = {cop.conditional_cdf(0.7, 0.3):.6f}')

        plot_dir = Path('./data/Plots/Copulas')
        plot_dir.mkdir(parents=True, exist_ok=True)
        cop.plot_pdf(plot_dir)
        cop.plot_cdf(plot_dir)
        cop.plot_overview(plot_dir)
