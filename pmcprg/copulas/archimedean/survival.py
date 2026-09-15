"""
Survival (180° rotation) copulas.

Given a base copula C, its survival copula is:
    Ĉ(u,v) = u + v − 1 + C(1−u, 1−v)
    ĉ(u,v) = c(1−u, 1−v)
    ĥ(v|u) = 1 − h_C(1−v | 1−u)
    (λ_L, λ_U)  of Ĉ = (λ_U, λ_L) of C   [tail roles swap]
    τ_K is preserved (concordance invariant under (u,v)→(1−u,1−v)).

Concrete subclasses: SurvivalClayton, SurvivalGH, SurvivalJoe.

Reference: Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed.,
Springer, ch. 2 (survival copulas, Ĉ(u,v) = u + v − 1 + C(1−u, 1−v)).

Numerics
--------
The wrapper used to call the base copula at ``(1.0 − u, 1.0 − v)`` formed
in floating point. For u = 10⁻¹² that is ``1 − 1.000089·10⁻¹²``: the
complement the base needs (log u for Joe, −log(1 − u) for GH and Clayton)
loses four digits, which put the log-densities of SGH and SJoe off by up
to 4·10⁻⁴ and 8·10⁻⁴ nat and SGH's h-function by 2.5·10⁻⁸ on the grid
of ``test_copula_limits.py`` (audit FR-1, roadmap technique 8). The base families now expose a *kernel interface* — the
complement is never formed, the kernel coordinate of 1 − u is computed from
u itself:

``_kcoord(x)`` / ``_kcoord_reflected(x)``
    kernel coordinate of x, and of 1 − x computed from x
    (``log x`` and ``log1p(−x)`` for Clayton and GH; the reverse for Joe);
``_k_logpdf(ka, kb)``
    log c(a, b);
``_k_cdf(ka, kb)``, ``_k_h(kb, ka)``
    ``(C, 1 − C)`` and ``(h(b|a), 1 − h(b|a))``, each member to relative
    precision;
``_k_inv_h(lw, ka)``
    ``(b, 1 − b)`` solving h(b | a) = w, from ``lw = log w``.

so that ĉ(u, v) = c from ``_kcoord_reflected(u), _kcoord_reflected(v)``,
ĥ(v|u) is the ``1 − h`` member, inv ĥ is the ``1 − b`` member of the base
inverse at ``lw = log1p(−w)``, and

    Ĉ(u, v) = (u − (1 − v)) + C     if C ≤ 1/2,
              (u + v) − (1 − C)     otherwise,

whose rounding error is ε·max(|u − (1 − v)|, C) or ε·max(u + v, 1 − C) —
proportional to Ĉ in both corners where it is small, instead of ε.
"""
if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import logging
import numpy as np

from pmcprg.copulas._base                   import CopulaVirt
from pmcprg.copulas.archimedean.clayton     import CopulaClayton
from pmcprg.copulas.archimedean.gumbel      import CopulaGH
from pmcprg.copulas.archimedean.joe         import CopulaJoe
from pmcprg.numerics                     import EPS, ONE_MINUS_EPS, minmaxEPS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generic wrapper
# ---------------------------------------------------------------------------

class SurvivalCopula(CopulaVirt):
    """180°-rotation wrapper around a base copula exposing the kernel interface.

    Subclasses must set the class attribute ``_base_class`` to a family that
    implements ``_kcoord_reflected``, ``_k_logpdf``, ``_k_cdf``, ``_k_h`` and
    ``_k_inv_h`` (module docstring).
    """

    _base_class: type[CopulaVirt] | None = None

    _KERNEL_INTERFACE = ('_kcoord_reflected', '_k_logpdf', '_k_cdf', '_k_h', '_k_inv_h')

    def __init__(self, **kwargs):
        missing = [m for m in self._KERNEL_INTERFACE if not hasattr(self._base_class, m)]
        if missing:
            raise TypeError(f"{type(self).__name__}: base {getattr(self._base_class, '__name__', None)} "
                            f"lacks the kernel interface {missing} (see pmcprg.copulas.archimedean.survival).")
        # Build base copula first; _update_params (called by super) will sync it.
        self._base: CopulaVirt = self._base_class(**kwargs)  # type: ignore[misc]
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    def _update_params(self):
        """Propagate current params to the base copula and mirror theta."""
        self._base.params = self.params
        self._base._update_params()
        if hasattr(self._base, 'theta'):
            self.theta = self._base.theta

    @staticmethod
    def _columns(uv):
        uv = np.asarray(uv, dtype=float)
        return (np.clip(uv[:, 0], EPS, ONE_MINUS_EPS),
                np.clip(uv[:, 1], EPS, ONE_MINUS_EPS))

    def _logpdf(self, u, v):
        b = self._base
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            return b._k_logpdf(b._kcoord_reflected(u), b._kcoord_reflected(v))

    def _cdf(self, u, v):
        b = self._base
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            c, cbar = b._k_cdf(b._kcoord_reflected(u), b._kcoord_reflected(v))
            out = np.where(c <= 0.5, (u - (1.0 - v)) + c, (u + v) - cbar)
        return np.clip(out, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Core functions — base kernel at the reflected coordinates
    # ------------------------------------------------------------------

    def pdf(self, uv):
        """ĉ(u,v) = c(1−u, 1−v)."""
        u = np.float64(minmaxEPS(uv[0]))
        v = np.float64(minmaxEPS(uv[1]))
        with np.errstate(over='ignore'):
            return float(np.exp(self._logpdf(u, v)))

    def pdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised ĉ(u,v) = c(1−u, 1−v), ``exp`` of :meth:`logpdf_array` (no floor)."""
        return np.exp(self.logpdf_array(uv))

    def logpdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised log ĉ(u,v) = log c(1−u, 1−v), complements never formed."""
        return self._logpdf(*self._columns(uv))

    def cdf(self, uv):
        """Ĉ(u,v) = u + v − 1 + C(1−u, 1−v)."""
        u = np.float64(minmaxEPS(uv[0]))
        v = np.float64(minmaxEPS(uv[1]))
        return float(self._cdf(u, v))

    def cdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised Ĉ(u,v) = u + v − 1 + C(1−u, 1−v) (module docstring)."""
        return self._cdf(*self._columns(uv))

    def conditional_cdf(self, v: float, u: float) -> float:
        """ĥ(v|u) = 1 − h_C(1−v | 1−u), the ``1 − h`` member of the base kernel."""
        b = self._base
        u = np.float64(minmaxEPS(u))
        v = np.float64(minmaxEPS(v))
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            _, hbar = b._k_h(b._kcoord_reflected(v), b._kcoord_reflected(u))
        return float(np.clip(hbar, 0.0, 1.0))

    def inv_h(self, w: float, u: float) -> float:
        """Inverse h-function via the survival relation.

        Solving ĥ(v|u) = w means ``1 − h_C(1−v | 1−u) = w``, i.e.
        ``h_C(1−v | 1−u) = 1−w``, so ``v = 1 − inv_h_base(1 − w, 1 − u)`` —
        taken as the ``1 − b`` member of the base kernel inverse, evaluated at
        ``log(1 − w) = log1p(−w)`` and the reflected coordinate of u.
        """
        return float(self.inv_h_array(np.array([w], dtype=float), np.array([u], dtype=float))[0])

    def inv_h_array(self, w: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Vectorised version: v = 1 − inv_h_base(1 − w, 1 − u), complements never formed."""
        b = self._base
        w = np.clip(np.asarray(w, dtype=float), EPS, ONE_MINUS_EPS)
        u = np.clip(np.asarray(u, dtype=float), EPS, ONE_MINUS_EPS)
        if w.shape != u.shape:
            raise ValueError(f"inv_h_array: w {w.shape} and u {u.shape} must match.")
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            _, vbar = b._k_inv_h(np.log1p(-w), b._kcoord_reflected(u))
        return np.clip(vbar, EPS, ONE_MINUS_EPS)

    def tail_dependence(self) -> tuple[float, float]:
        """Swap λ_L and λ_U of the base copula."""
        lam_L, lam_U = self._base.tail_dependence()
        return lam_U, lam_L


# ---------------------------------------------------------------------------
# Concrete survival copulas
# ---------------------------------------------------------------------------

class SurvivalClayton(SurvivalCopula):
    """Survival Clayton copula — upper-tail dependence.

    Base: Clayton (lower-tail) → rotation gives upper-tail.
    λ_L = 0, λ_U = 2^{−1/θ_Clayton}.
    """
    _base_class = CopulaClayton


class SurvivalGH(SurvivalCopula):
    """Survival Gumbel-Hougaard copula — lower-tail dependence.

    Base: GH (upper-tail) → rotation gives lower-tail.
    λ_L = 2 − 2^{1/θ_GH}, λ_U = 0.
    """
    _base_class = CopulaGH


class SurvivalJoe(SurvivalCopula):
    """Survival Joe copula — lower-tail dependence.

    Base: Joe (upper-tail) → rotation gives lower-tail.
    λ_L = 2 − 2^{1/θ_Joe}, λ_U = 0.
    """
    _base_class = CopulaJoe


# ---------------------------------------------------------------------------
# Quick smoke test / demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from pathlib import Path

    for cls, name in [
        (SurvivalClayton, 'SurvivalClayton'),
        (SurvivalGH,      'SurvivalGH'),
        (SurvivalJoe,     'SurvivalJoe'),
    ]:
        cop = cls(tau_k=0.5)
        lam_L, lam_U = cop.tail_dependence()
        print(f'\n--- {name} ---')
        print(f'tau_k    : {cop.params["tau_k"]:.4f}  range={cop.tau_range}')
        print(f'theta    : {cop.theta:.6f}')
        print(f'pdf(0.3, 0.7) = {cop.pdf([0.3, 0.7]):.6f}')
        print(f'cdf(0.3, 0.7) = {cop.cdf([0.3, 0.7]):.6f}')
        print(f'h(0.7 | 0.3)  = {cop.conditional_cdf(0.7, 0.3):.6f}')
        print(f'tail dep : λ_L = {lam_L:.4f},  λ_U = {lam_U:.4f}')

        plot_dir = Path('./data/Plots/Copulas')
        plot_dir.mkdir(parents=True, exist_ok=True)
        cop.plot_pdf(plot_dir)
        cop.plot_cdf(plot_dir)
        cop.plot_overview(plot_dir)
