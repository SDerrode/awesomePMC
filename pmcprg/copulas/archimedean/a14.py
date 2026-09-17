"""
Archimedean copula A14 — Nelsen (2006) Table 4.1, family (4.2.14); numbered
14 in Huard, Évin & Favre (2006), Table 1, who reuse Nelsen's numbering.

Generator: φ(t) = (t^{−1/θ} − 1)^θ,  θ ≥ 1.
CDF:  C(u,v) = (1 + (U1 + U2)^{1/θ})^{-θ}
      where  U_i = (u_i^{-1/θ} − 1)^θ

Kendall's τ is  τ = 1 − 2/(1 + 2θ)  (Nelsen, family 14), so
θ = 1/(1 − τ_K) − 1/2,  τ_K ∈ [1/3, 1),  θ ≥ 1.
λ_L = 1/2 for every θ  (C(u,u) = (1 + 2^{1/θ}(u^{−1/θ} − 1))^{−θ} → u/2),
λ_U = 2 − 2^{1/θ}.

Numerics
--------
``U_i`` is exponentially small by construction — 7·10⁻²⁹ at u = 0.5 for
θ = 19.5 (τ = 0.95) — so it is never formed: every path works on
``log U_i = θ · log(expm1(−log u / θ))`` and ``log S = logaddexp(log U1,
log U2)``. The former ``np.maximum(U_i, EPS)`` floor in ``logpdf_array``
overstated log c by 0.77 nat at the centre of the square and by ~100 nat in
the corners at τ = 0.95 (audit K-2). Nothing overflows for u in the open
square, so no path has a fallback any more: the scalar pdf returned ``EPS``,
``cdf_array`` ``EPS`` and h a 0/1 step on a non-finite value; a value that
cannot be computed is now NaN, and a density that underflows is 0.0
(audit RB-9). Log-space evaluation of Archimedean densities: Hofert,
Mächler & McNeil (2012), *J. Multivariate Anal.* 110, 133–150,
doi:10.1016/j.jmva.2012.02.019.

Kernel interface (audit FR-8, A12/A14 round). Every quantity above is a
function of ``log u`` alone — ``a = u^{−1/θ} − 1 = expm1(−log u/θ)``,
``1 − u^{1/θ} = −expm1(log u/θ)`` — so A14's natural kernel coordinate is
``ka = log u``, the same as GH's and BB1's (its generator
``(t^{−1/θ} − 1)^θ`` is built from ``t`` directly, not from ``1 − t`` as
Joe's). ``_kcoord_reflected(x) = log1p(−x)`` is the exact log of ``1 − x``
for the 90°/270° rotation wrapper (``pmcprg.copulas.archimedean.rotated``).
The refactor only replaces ``np.log(u)`` by ``ka``: every public value is
unchanged, bit for bit.

In ``a``, h(v|u) = (a/t)^{θ−1} ((1 + a)/(1 + t))^{θ+1} with
``t = (a^θ + b^θ)^{1/θ}`` and ``1 + a = u^{−1/θ}`` — A12's shape with the
exponent 2 replaced by θ + 1 (module docstring of ``a12.py``) — so ``_k_h``'s
``1 − h`` member and ``_k_inv_h`` (monotone Newton, then
``v = (1 + b)^{−θ}``) reuse A12's two helpers with p = θ + 1. The public
``inv_h``/``inv_h_array`` remain :class:`CopulaVirt`'s.
"""
if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import logging
import numpy as np

from pmcprg.copulas._base import CopulaVirt
from pmcprg.copulas.archimedean.a12 import _nelsen_inv_h_logb, _nelsen_logh_nocancel
from pmcprg.numerics   import EPS, ONE_MINUS_EPS, minmaxEPS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Log-space kernel on ka = log u, kb = log v (module docstring) — shared by
# every evaluation path, including the rotation wrapper's kernel interface.
# ---------------------------------------------------------------------------

def _a14_loga(ka, th):
    """``log a`` with ``a = u^{−1/θ} − 1 = U^{1/θ}``, from ``ka = log u``."""
    return np.log(np.expm1(-ka / th))


def _a14_logu(ka, th):
    """``log U`` with ``U = (u^{−1/θ} − 1)^θ``, without forming ``U``."""
    return th * np.log(np.expm1(-ka / th))


def _a14_terms(ka, kb, th):
    """``(log U1, log U2, log S, S^{1/θ})`` from ``ka = log u``, ``kb = log v``."""
    log_u1 = _a14_logu(ka, th)
    log_u2 = _a14_logu(kb, th)
    log_s  = np.logaddexp(log_u1, log_u2)
    s1t    = np.exp(log_s / th)
    return log_u1, log_u2, log_s, s1t


def _a14_logpdf(ka, kb, th):
    """log c(u, v) from ``ka = log u``, ``kb = log v``.

    c = U1 U2 S^{1/θ−2} (1 + S^{1/θ})^{−2−θ} (θ − 1 + 2θ S^{1/θ})
        / [θ u v (1 − u^{1/θ})(1 − v^{1/θ})],
    with ``log(1 − u^{1/θ}) = log(−expm1(log u / θ))`` — stable as u → 1.
    """
    log_u1, log_u2, log_s, s1t = _a14_terms(ka, kb, th)
    with np.errstate(divide='ignore'):
        # θ − 1 + 2θ S^{1/θ}: at θ = 1 the constant vanishes and the term is
        # 2 S^{1/θ}; logaddexp(−inf, x) = x handles that boundary exactly.
        log_last = np.logaddexp(np.log(max(th - 1.0, 0.0)),
                                np.log(2.0 * th) + log_s / th)
    return (log_u1 + log_u2
            + (1.0 / th - 2.0) * log_s
            - (2.0 + th) * np.log1p(s1t)
            + log_last
            - np.log(th) - ka - kb
            - np.log(-np.expm1(ka / th))
            - np.log(-np.expm1(kb / th)))


def _a14_cdf(ka, kb, th):
    """``(C, 1 − C)``: C = (1 + S^{1/θ})^{−θ} = exp(−θ log1p(S^{1/θ}))."""
    _, _, _, s1t = _a14_terms(ka, kb, th)
    e = -th * np.log1p(s1t)
    return np.exp(e), -np.expm1(e)


def _a14_h(kb, ka, th):
    """``(h, 1 − h)`` of h(v|u) = (1+S^{1/θ})^{−θ−1} · S^{1/θ−1} · (u^{−1/θ}−1)^{θ−1} · u^{−1/θ−1}.

    ``(u^{−1/θ} − 1)^{θ−1} = U1^{(θ−1)/θ}``, so everything is in log space.
    h is that textbook product (unchanged, so the public
    :meth:`CopulaA14.conditional_cdf` keeps its values bit for bit); 1 − h
    comes from the cancellation-free form shared with A12
    (:func:`pmcprg.copulas.archimedean.a12._nelsen_logh_nocancel`) with
    ``a = u^{−1/θ} − 1``, ``r = a/(1 + a) = 1 − u^{1/θ}`` (``log(1 + a) =
    −log u/θ`` exactly) and p = θ + 1.
    """
    log_u1, _, log_s, s1t = _a14_terms(ka, kb, th)
    logh = (-(th + 1.0) * np.log1p(s1t)
            + (1.0 / th - 1.0) * log_s
            + ((th - 1.0) / th) * log_u1
            - (1.0 / th + 1.0) * ka)
    la = _a14_loga(ka, th)
    logh_nc = _nelsen_logh_nocancel(la, _a14_loga(kb, th), la + ka / th, th, th + 1.0)
    return np.exp(logh), -np.expm1(logh_nc)


def _a14_inv_h(lw, ka, th):
    """``(v, 1 − v)`` solving h(v|u) = w, from ``lw = log w`` and ``ka = log u``.

    Monotone Newton (:func:`pmcprg.copulas.archimedean.a12._nelsen_inv_h_logb`)
    for ``log b``, b = v^{−1/θ} − 1; then ``log v = −θ·log(1 + b)``, and
    ``1 − v = −expm1(log v)`` without forming the complement.
    """
    la = _a14_loga(ka, th)
    lb = _nelsen_inv_h_logb(lw, la, la + ka / th, th, th + 1.0)
    lv = -th * np.logaddexp(0.0, lb)
    return np.exp(lv), -np.expm1(lv)


class CopulaA14(CopulaVirt):

    def __init__(self, **kwargs):
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    def _update_params(self):
        # τ = 1 − 2/(1+2θ)  ⟹  θ = 1/(1−τ) − 1/2.
        # The former expression, 2/(3(1−τ)), agrees with this one *only* at
        # τ = 1/3 (both give θ = 1), which is the family's lower τ bound — so
        # every fixture pinned at the boundary looked correct while the copula
        # silently realised τ = 0.539 when asked for 0.6.
        self.theta = 1.0 / (1.0 - self.params['tau_k']) - 0.5

    # ------------------------------------------------------------------
    # Kernel interface (audit FR-8) — consumed by the 90°/270° rotation
    # wrapper in ``pmcprg.copulas.archimedean.rotated``. The kernel
    # coordinate of a point x is log x (module docstring), as for GH/BB1:
    # ``_kcoord_reflected(x)`` computes it for 1 − x, i.e. log1p(−x).
    # ------------------------------------------------------------------

    @staticmethod
    def _kcoord(x):
        return np.log(x)

    @staticmethod
    def _kcoord_reflected(x):
        return np.log1p(-x)

    def _k_logpdf(self, ka, kb):
        return _a14_logpdf(ka, kb, self.theta)

    def _k_cdf(self, ka, kb):
        return _a14_cdf(ka, kb, self.theta)

    def _k_h(self, kb, ka):
        return _a14_h(kb, ka, self.theta)

    def _k_inv_h(self, lw, ka):
        return _a14_inv_h(lw, ka, self.theta)

    # ------------------------------------------------------------------
    # Public API — built from the kernel interface above
    # ------------------------------------------------------------------

    def pdf(self, uv):
        """c(u, v) = exp(log c), see :func:`_a14_logpdf`; 0.0 only on underflow."""
        u0 = minmaxEPS(uv[0])
        u1 = minmaxEPS(uv[1])
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            return float(np.exp(self._k_logpdf(self._kcoord(np.float64(u0)),
                                               self._kcoord(np.float64(u1)))))

    def cdf(self, uv):
        u0 = minmaxEPS(uv[0])
        u1 = minmaxEPS(uv[1])
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            c, _ = self._k_cdf(self._kcoord(np.float64(u0)), self._kcoord(np.float64(u1)))
            return float(np.clip(c, 0.0, 1.0))

    def cdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised closed-form CDF (A14 has no statsmodels backend)."""
        uv = np.asarray(uv, dtype=float)
        u  = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v  = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            c, _ = self._k_cdf(self._kcoord(u), self._kcoord(v))
            return np.clip(c, 0.0, 1.0)

    def pdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised closed-form PDF, ``exp`` of :meth:`logpdf_array`."""
        return np.exp(self.logpdf_array(uv))

    def logpdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Native log-PDF for A14, computed entirely in log space.

        With U_i = (u_i^{−1/θ} − 1)^θ > 0, S = U_1 + U_2 > 0:

            log c = log U_1 + log U_2 + (1/θ − 2) log S + (−2 − θ) log(1 + S^{1/θ})
                  − log(θ uv) − log(1 − u^{1/θ}) − log(1 − v^{1/θ})
                  + log(θ − 1 + 2θ S^{1/θ}).
        """
        uv = np.asarray(uv, dtype=float)
        u  = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v  = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            return self._k_logpdf(self._kcoord(u), self._kcoord(v))

    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = (1+S^{1/θ})^{−θ−1} · S^{1/θ−1} · (u^{−1/θ}−1)^{θ−1} · u^{−1/θ−1}."""
        u = minmaxEPS(u)
        v = minmaxEPS(v)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            h, _ = self._k_h(self._kcoord(np.float64(v)), self._kcoord(np.float64(u)))
            return float(np.clip(h, 0.0, 1.0))

    def tail_dependence(self) -> tuple[float, float]:
        """A14: λ_L = 1/2 for every θ, λ_U = 2 − 2^{1/θ} (Nelsen 2006, family 4.2.14).

        Closed form. The inherited numeric limit at u = 10⁻⁴ converges like
        u^{1/θ} and returned λ_L = 0.65 / 0.77 / 0.94 at τ = 0.9 / 0.95 / 0.99.
        """
        return 0.5, float(2.0 - 2.0 ** (1.0 / self.theta))

    # Numerical majorant and the public inv_h/inv_h_array (Brent on
    # conditional_cdf) are inherited from CopulaVirt, unchanged by the
    # kernel refactor; ``_k_inv_h`` serves the rotations only.


if __name__ == '__main__':
    from pathlib import Path

    cop = CopulaA14(tau_k=0.5)
    print(f'Copula   : {cop.copula_enum.value.LONG_NAME}')
    print(f'tau_k    : {cop.params["tau_k"]:.4f}  range={cop.tau_range}')
    print(f'theta    : {cop.theta:.6f}  [= 1/(1−τ) − 1/2]')
    print(f'pdf(0.3, 0.7) = {cop.pdf([0.3, 0.7]):.6f}')
    print(f'cdf(0.3, 0.7) = {cop.cdf([0.3, 0.7]):.6f}')
    print(f'h(0.7 | 0.3)  = {cop.conditional_cdf(0.7, 0.3):.6f}')
    lL, lU = cop.tail_dependence()
    print(f'tail dep : λ_L = {lL:.4f}  [= 1/2 for any θ],  λ_U = {lU:.4f}  [= 2 − 2^(1/θ)]')

    plot_dir = Path('./data/Plots/Copulas')
    plot_dir.mkdir(parents=True, exist_ok=True)
    cop.plot_pdf(plot_dir)
    cop.plot_cdf(plot_dir)
    cop.plot_h_function(plot_dir)
    cop.plot_samples(plot_dir)
    cop.plot_overview(plot_dir)
    cop.plot_multi_tau(plot_dir)
