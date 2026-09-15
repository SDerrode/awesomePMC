"""
Clayton copula — lower-tail dependence.

Generator: φ(t) = (t^{−θ} − 1)/θ,  θ = 2τ_K/(1 − τ_K) > 0.
CDF:  C(u,v) = (u^{−θ} + v^{−θ} − 1)^{−1/θ}
PDF:  c(u,v) = (1+θ) (uv)^{−1−θ} (u^{−θ} + v^{−θ} − 1)^{−1/θ−2}
h:    h(v|u) = C(u,v)^{1+θ} · u^{−θ−1}
τ = θ/(θ+2);  λ_L = 2^{−1/θ},  λ_U = 0.
θ → 0 → independence;  θ → ∞ → comonotonicity.

Reference: Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed.,
Springer, Table 4.1 family (4.2.1) and ch. 5 (τ); Joe, H. (1997).
*Multivariate Models and Dependence Concepts*, Chapman & Hall, ch. 5
(one-parameter families) and ch. 2 (tail dependence).

Numerics
--------
Two defects of the textbook expressions (audit RB-1, FR-1):

* near independence, ``u^{−θ} + v^{−θ} − 1`` is 1 + O(θ) and its rounding
  error is multiplied by ``(2 + 1/θ)``: at θ = 10⁻¹⁵ the log-likelihood of
  200 independent uniforms came out at +0.58 instead of −1.5·10⁻¹⁶, and the
  CDF off by 4·10⁻⁵ at θ = 2·10⁻¹²;
* in strong dependence ``u^{−θ}`` overflows below u = exp(−709.78/θ),
  9.8·10⁻¹² at θ = 28 — the statsmodels backend this module delegated to
  returned NaN there, and the scalar paths an ``EPS`` sentinel.

Every path now works on the kernel coordinates ``ka = log a``, ``kb = log b``
(``log1p(−u)`` when the survival copula evaluates the base at 1 − u, see
``survival.py``) through ``x = −θ ka ≥ 0``, ``y = −θ kb ≥ 0``,
``m = max(x, y)``, ``n = min(x, y)``::

    L = log(a^{−θ} + b^{−θ} − 1) = m + λ,   λ = log1p(e^{n−m} · (−expm1(−n)))

the log-space evaluation of Archimedean densities of Hofert, Mächler &
McNeil (2012), *J. Multivariate Anal.* 110, 133–150,
doi:10.1016/j.jmva.2012.02.019, as R ``copula`` applies it to Clayton
(roadmap technique 4), with its two branches — ``log1p(expm1(x) + expm1(y))``
for small θ|log u|, ``logaddexp(x, y) + log1mexp(·)`` for large θ|log u| —
merged into one expression: ``e^m + e^n − 1 = e^m(1 + e^{n−m}(1 − e^{−n}))``.
Both terms are non-negative, so L keeps its relative precision as θ → 0,
and nothing overflows for any θ. Then::

    log c = log1p(θ) − (1 + θ)(ka + kb) − (2 + 1/θ)·L
    C     = exp(−L/θ),                     1 − C = −expm1(−L/θ)
    log h = −(1 + 1/θ)·((m − x) + λ)       [h(b | a) = C^{1+θ} a^{−1−θ}; 1 − h = −expm1(log h)]
    inv_h : y = logaddexp(0, x + log expm1(G)),  G = −θ log w/(1 + θ)

(``log h`` is ``−(1 + 1/θ)(L − x)`` with ``L − x`` written as two
non-negative terms; ``log expm1(G) = G + log(−expm1(−G))``.) The residual
cancellation in log c between ``−(1+θ)(ka + kb)`` and ``−L/θ`` is absolute,
≈ ε·|ka + kb| ≤ 10⁻¹⁴ nat, not ε/θ. θ = 0 itself (τ = 0) is outside the
registered τ range.
"""
if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import logging
import numpy as np

from pmcprg.copulas._base import CopulaVirt
from pmcprg.numerics   import EPS, ONE_MINUS_EPS, minmaxEPS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Log-space kernel on ka = log a, kb = log b (module docstring)
# ---------------------------------------------------------------------------

def _clayton_l(ka, kb, th):
    """``(x, m, λ)`` with ``x = −θ ka`` and ``L = m + λ``."""
    x = -th * ka
    y = -th * kb
    m = np.maximum(x, y)
    n = np.minimum(x, y)
    return x, m, np.log1p(np.exp(n - m) * -np.expm1(-n))


def _clayton_logpdf(ka, kb, th):
    _, m, lam = _clayton_l(ka, kb, th)
    return np.log1p(th) - (1.0 + th) * (ka + kb) - (2.0 + 1.0 / th) * (m + lam)


def _clayton_cdf(ka, kb, th):
    """``(C, 1 − C)``."""
    _, m, lam = _clayton_l(ka, kb, th)
    e = -(m + lam) / th
    return np.exp(e), -np.expm1(e)


def _clayton_h(kb, ka, th):
    """``(h, 1 − h)`` of h(b | a)."""
    x, m, lam = _clayton_l(ka, kb, th)
    logh = -(1.0 + 1.0 / th) * ((m - x) + lam)
    return np.exp(logh), -np.expm1(logh)


def _clayton_inv_h(lw, ka, th):
    """``(b, 1 − b)`` solving h(b | a) = w, from ``lw = log w`` and ``ka = log a``."""
    g = -lw * (th / (1.0 + th))
    log_expm1_g = g + np.log(-np.expm1(-g))
    y = np.logaddexp(0.0, -th * ka + log_expm1_g)
    kb = -y / th
    return np.exp(kb), -np.expm1(kb)


class CopulaClayton(CopulaVirt):
    """
    Clayton copula — lower-tail dependence.

    θ = 2 τ_K / (1 − τ_K).
    """

    def __init__(self, **kwargs):
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    def _update_params(self):
        tau = self.params['tau_k']
        self.theta = 2.0 * tau / (1.0 - tau)

    # -- kernel interface (consumed by survival.py) -------------------------
    # The kernel coordinate of a point x is log x: ``_kcoord(x)`` computes it
    # for x itself, ``_kcoord_reflected(x)`` for 1 − x — which is log1p(−x).

    @staticmethod
    def _kcoord(x):
        return np.log(x)

    @staticmethod
    def _kcoord_reflected(x):
        return np.log1p(-x)

    def _k_logpdf(self, ka, kb):
        return _clayton_logpdf(ka, kb, self.theta)

    def _k_cdf(self, ka, kb):
        return _clayton_cdf(ka, kb, self.theta)

    def _k_h(self, kb, ka):
        return _clayton_h(kb, ka, self.theta)

    def _k_inv_h(self, lw, ka):
        return _clayton_inv_h(lw, ka, self.theta)

    # -- public API ----------------------------------------------------------

    def pdf(self, uv):
        u = np.float64(minmaxEPS(uv[0]))
        v = np.float64(minmaxEPS(uv[1]))
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            return float(np.exp(_clayton_logpdf(np.log(u), np.log(v), self.theta)))

    def pdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised closed-form PDF, ``exp`` of :meth:`logpdf_array` (no floor)."""
        return np.exp(self.logpdf_array(uv))

    def logpdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Native log-PDF in log space (module docstring).

            log c(u, v) = log(1 + θ) − (1 + θ)(log u + log v) − (2 + 1/θ)·L
        """
        uv = np.asarray(uv, dtype=float)
        u  = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v  = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            return _clayton_logpdf(np.log(u), np.log(v), self.theta)

    def cdf(self, uv):
        u = np.float64(minmaxEPS(uv[0]))
        v = np.float64(minmaxEPS(uv[1]))
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            c, _ = _clayton_cdf(np.log(u), np.log(v), self.theta)
        return float(np.clip(c, 0.0, 1.0))

    def cdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised closed-form CDF (NaN where it cannot be evaluated)."""
        uv = np.asarray(uv, dtype=float)
        u  = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v  = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            c, _ = _clayton_cdf(np.log(u), np.log(v), self.theta)
        return np.clip(c, 0.0, 1.0)

    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = C(u,v)^{1+θ} · u^{−θ−1}, in log space (module docstring).
        Pour u → 0+, Clayton a une dépendance de queue inférieure : h(v|u) → 1.
        """
        u = np.float64(minmaxEPS(u))
        v = np.float64(minmaxEPS(v))
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            h, _ = _clayton_h(np.log(v), np.log(u), self.theta)
        return float(np.clip(h, 0.0, 1.0))

    def inv_h(self, w: float, u: float) -> float:
        """Closed-form inverse of h(v|u): solve h(v|u) = w analytically.

        From h(v|u) = (1 + u^θ·(v^{−θ} − 1))^{−1−1/θ} = w:
            v = ((w^{−θ/(1+θ)} − 1)·u^{−θ} + 1)^{−1/θ},
        evaluated in log space (module docstring) — no overflow of u^{−θ}.
        """
        u = np.float64(minmaxEPS(u))
        w = np.float64(minmaxEPS(w))
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            v, _ = _clayton_inv_h(np.log(w), np.log(u), self.theta)
        return minmaxEPS(float(v)) if np.isfinite(v) else float('nan')

    def inv_h_array(self, w: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Vectorised closed-form inverse h-function (NaN where it cannot be evaluated)."""
        w  = np.clip(np.asarray(w, dtype=float), EPS, ONE_MINUS_EPS)
        u  = np.clip(np.asarray(u, dtype=float), EPS, ONE_MINUS_EPS)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            v, _ = _clayton_inv_h(np.log(w), np.log(u), self.theta)
        return np.clip(v, EPS, ONE_MINUS_EPS)

    def tail_dependence(self) -> tuple[float, float]:
        """Clayton copula: λ_L = 2^{−1/θ}, λ_U = 0."""
        return 2.0 ** (-1.0 / self.theta), 0.0

    def majorant(self, u_left: float) -> float:
        """Closed-form maximum of c(u, v) over v.

        Derivation
        ----------
        With ``c(u,v) = (1+θ) (uv)^{-1-θ} (u^{-θ}+v^{-θ}-1)^{-1/θ-2}``,
        setting ∂/∂v log c = 0 gives the critical value

            v*^{-θ} = (1+θ)/θ · (u^{-θ} − 1),

        i.e. ``log v* = −[log((1+θ)/θ) + log expm1(x)]/θ`` with ``x = −θ log u``,
        evaluated without forming u^{−θ}. If v* ≥ 1 the interior critical point
        does not exist and the max occurs at a boundary; the density is also
        evaluated at both boundaries ``v = EPS`` and ``v = 1 − EPS`` and the
        largest of the candidates is returned.
        """
        u = minmaxEPS(u_left)
        th = self.theta
        x = -th * np.log(u)
        with np.errstate(divide='ignore'):
            log_v_star = -(np.log1p(1.0 / th) + x + np.log(-np.expm1(-x))) / th
        if log_v_star < 0.0:
            f_star = self.pdf([u, minmaxEPS(float(np.exp(log_v_star)))])
        else:
            f_star = 0.0
        f_lo = self.pdf([u, EPS])
        f_hi = self.pdf([u, ONE_MINUS_EPS])
        return float(max(f_star, f_lo, f_hi))


if __name__ == '__main__':
    from pathlib import Path

    cop = CopulaClayton(tau_k=0.5)
    lam_l = 2.0 ** (-1.0 / cop.theta)
    print(f'Copula   : {cop.copula_enum.value.LONG_NAME}')
    print(f'tau_k    : {cop.params["tau_k"]:.4f}  range={cop.tau_range}')
    print(f'theta    : {cop.theta:.6f}  [= 2τ/(1−τ)]')
    print(f'pdf(0.3, 0.7) = {cop.pdf([0.3, 0.7]):.6f}')
    print(f'cdf(0.3, 0.7) = {cop.cdf([0.3, 0.7]):.6f}')
    print(f'h(0.7 | 0.3)  = {cop.conditional_cdf(0.7, 0.3):.6f}')
    print(f'tail dep : λ_L = {lam_l:.4f}  [= 2^(−1/θ)],  λ_U = 0')

    plot_dir = Path('./data/Plots/Copulas')
    plot_dir.mkdir(parents=True, exist_ok=True)
    cop.plot_pdf(plot_dir)
    cop.plot_cdf(plot_dir)
    cop.plot_h_function(plot_dir)
    cop.plot_samples(plot_dir)
    cop.plot_overview(plot_dir)
    cop.plot_multi_tau(plot_dir)
