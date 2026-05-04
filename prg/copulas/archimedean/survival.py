"""
Survival (180° rotation) copulas.

Given a base copula C, its survival copula is:
    Ĉ(u,v) = u + v − 1 + C(1−u, 1−v)
    ĉ(u,v) = c(1−u, 1−v)
    ĥ(v|u) = 1 − h_C(1−v | 1−u)
    (λ_L, λ_U)  of Ĉ = (λ_U, λ_L) of C   [tail roles swap]
    τ_K is preserved (concordance invariant under (u,v)→(1−u,1−v)).

Concrete subclasses: SurvivalClayton, SurvivalGH, SurvivalJoe.
"""
if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import logging
import numpy as np

from prg.copulas._base                   import CopulaVirt
from prg.copulas.archimedean.clayton     import CopulaClayton
from prg.copulas.archimedean.gumbel      import CopulaGH
from prg.copulas.archimedean.joe         import CopulaJoe
from prg.numerics                     import EPS, ONE_MINUS_EPS, minmaxEPS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generic wrapper
# ---------------------------------------------------------------------------

class SurvivalCopula(CopulaVirt):
    """180°-rotation wrapper around any CopulaVirt subclass.

    Subclasses must set the class attribute ``_base_class``.
    """

    _base_class: type[CopulaVirt] | None = None

    def __init__(self, **kwargs):
        # Build base copula first; _update_params (called by super) will sync it.
        self._base: CopulaVirt = self._base_class(**kwargs)  # type: ignore[misc]
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    def _update_params(self):
        """Propagate current params to the base copula and mirror theta."""
        self._base.params = self.params
        self._base._update_params()
        if hasattr(self._base, 'theta'):
            self.theta = self._base.theta

    # ------------------------------------------------------------------
    # Core functions — delegate to base with (1−u, 1−v) substitution
    # ------------------------------------------------------------------

    def pdf(self, uv):
        """ĉ(u,v) = c(1−u, 1−v)."""
        u = minmaxEPS(uv[0])
        v = minmaxEPS(uv[1])
        return self._base.pdf([1.0 - u, 1.0 - v])

    def pdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised: ĉ(u,v) = c(1−u, 1−v) — delegate to the base copula."""
        uv = np.asarray(uv, dtype=float)
        return self._base.pdf_array(1.0 - uv)

    def logpdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised: log ĉ(u,v) = log c(1−u, 1−v) — delegate to the base copula."""
        uv = np.asarray(uv, dtype=float)
        return self._base.logpdf_array(1.0 - uv)

    def cdf(self, uv):
        """Ĉ(u,v) = u + v − 1 + C(1−u, 1−v)."""
        u = minmaxEPS(uv[0])
        v = minmaxEPS(uv[1])
        result = float(u) + float(v) - 1.0 + self._base.cdf([1.0 - u, 1.0 - v])
        return float(np.clip(result, 0.0, 1.0))

    def cdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised: Ĉ(u,v) = u + v − 1 + C(1−u, 1−v)."""
        uv = np.asarray(uv, dtype=float)
        u  = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v  = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        base_uv = np.column_stack((1.0 - u, 1.0 - v))
        return np.clip(u + v - 1.0 + self._base.cdf_array(base_uv), 0.0, 1.0)

    def conditional_cdf(self, v: float, u: float) -> float:
        """ĥ(v|u) = 1 − h_C(1−v | 1−u)."""
        u = minmaxEPS(u)
        v = minmaxEPS(v)
        return float(np.clip(1.0 - self._base.conditional_cdf(1.0 - v, 1.0 - u), 0.0, 1.0))

    def inv_h(self, w: float, u: float) -> float:
        """Inverse h-function via the survival relation.

        Solving ĥ(v|u) = w means ``1 − h_C(1−v | 1−u) = w``, i.e.
        ``h_C(1−v | 1−u) = 1−w``, so::

            v = 1 − inv_h_base(1 − w, 1 − u)

        Inherits the base copula's closed-form fast path when available
        (Gaussian / Clayton / Frank).
        """
        u = minmaxEPS(u)
        w = minmaxEPS(w)
        v = 1.0 - self._base.inv_h(1.0 - w, 1.0 - u)
        return minmaxEPS(v)

    def inv_h_array(self, w: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Vectorised version: v = 1 − inv_h_base_array(1 − w, 1 − u)."""
        w = np.clip(np.asarray(w, dtype=float), EPS, ONE_MINUS_EPS)
        u = np.clip(np.asarray(u, dtype=float), EPS, ONE_MINUS_EPS)
        return np.clip(1.0 - self._base.inv_h_array(1.0 - w, 1.0 - u),
                       EPS, ONE_MINUS_EPS)

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
