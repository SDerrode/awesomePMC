"""
90°/270°-rotation copulas (audit FR-8, Clayton pilot).

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

Concrete subclasses (this round, FR-8 pilot): ``CopulaClayton90``,
``CopulaClayton270``. GH, Joe, BB1, A12 and A14 are explicitly out of scope
this round; GH and Joe already expose the kernel interface (used by
``SurvivalGH``/``SurvivalJoe``) and can be added as two-line subclasses, same
as Clayton below. BB1, A12 and A14 do not yet expose it — that is the
prerequisite for a fast follow, not a defect of this mechanism (see the
module docstring of ``survival.py``, which stopped at the same three
families for the same reason).

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
from pmcprg.copulas.archimedean.clayton import CopulaClayton
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


if __name__ == '__main__':
    from pathlib import Path

    for cls, name in [
        (CopulaClayton90,  'CopulaClayton90'),
        (CopulaClayton270, 'CopulaClayton270'),
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
