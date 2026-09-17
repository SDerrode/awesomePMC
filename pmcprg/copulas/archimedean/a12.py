"""
Archimedean copula A12 — Nelsen (2006) Table 4.1, family (4.2.12); numbered
12 in Huard, Évin & Favre (2006), Table 1, who reuse Nelsen's numbering.

Generator: φ(t) = (1/t − 1)^θ,  θ ≥ 1.
CDF:       C(u,v) = 1 / (1 + (U1 + U2)^{1/θ})
           where  U_i = (1/u_i − 1)^θ
τ = 1 − 2/(3θ)  (Nelsen, family 12), so  θ = 2/(3(1 − τ_K)),  τ_K ∈ [1/3, 1).
λ_L = 2^{−1/θ},  λ_U = 2 − 2^{1/θ}.

Reference: Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed.,
Springer, Table 4.1 family (4.2.12) and ch. 5 (τ); Huard, D., Évin, G. &
Favre, A.-C. (2006). Bayesian copula selection. *Computational Statistics
& Data Analysis* 51(2), 809–822, Table 1 (secondary source, numbering).

Numerics
--------
The textbook ``1/u − 1`` cancels as u → 1: at u = 1 − 10⁻¹² it keeps four
significant digits, which put log c off by 1.2·10⁻⁴ nat at θ = 10/9 and
7·10⁻⁴ nat at θ = 20/3, and h(v|u) off by 10⁻⁸ (audit FR-1). The base is
written x = (1 − u)/u instead — 1 − u is exact in floating point for u ≥ ½
(Sterbenz) and ``log1p(−u)`` is accurate for u < ½ — and no power is ever
formed: every path works on

    log U_i = θ (log1p(−u_i) − log u_i),   log S = logaddexp(log U1, log U2),
    t = S^{1/θ} = exp(log S / θ),          log(1 + t) = logaddexp(0, log S / θ),

the log-space evaluation of Archimedean densities of Hofert, Mächler &
McNeil (2012), *J. Multivariate Anal.* 110, 133–150,
doi:10.1016/j.jmva.2012.02.019. With the derivatives of ψ(s) = 1/(1 + s^{1/θ}),

    c(u,v)  = x^{θ−1} y^{θ−1} S^{1/θ−2} (1+t)^{−3} [(θ−1) + (θ+1) t] / (u² v²),
    h(v|u)  = x^{θ−1} S^{1/θ−1} (1+t)^{−2} / u²,

with x^{θ−1}/u² = U1 / (u (1 − u)). Nothing overflows for u in the open
square at any θ, so there is no fallback: a value that cannot be computed is
NaN, not ``EPS`` or a 0/1 step (audit RB-9).

Kernel interface (audit FR-8, A12/A14 round). The rotation wrapper
(``pmcprg.copulas.archimedean.rotated``) needs every path to take a *kernel
coordinate* of each argument from which the coordinate of its complement is
available without forming ``1 − u``. A12's generator is built from
``(1 − t)/t``, and every formula above needs ``log u`` and ``log(1 − u)``
*separately* (the density's ``(θ − 1) log(1 − u) − (θ + 1) log u``, h's
``− log u − log(1 − u)``), so the natural coordinate is the **pair**
``ka = (log u, log(1 − u))`` — neither GH's ``log u`` nor Joe's
``log(1 − u)`` alone. Its reflection is the swapped pair, exact in floating
point (and ``log x`` of 1 − u is ``−log x`` of u). A single scalar such as
``log x`` would have served too (``log u = −softplus(log x)``), but only by
recomputing ``log u`` differently from the pre-refactor code, and the
refactor keeps every public value bit for bit; the wrapper treats the
coordinate as opaque, so a pair costs nothing there.

A12 and A14 share the shape of their h-function: in their own transformed
coordinates ``a`` of u and ``b`` of v (here ``a = x``, ``b = y``),

    h(v|u) = (a/t)^{θ−1} ((1 + a)/(1 + t))^p,   t = (a^θ + b^θ)^{1/θ},

with p = 2 here (``1 + x = 1/u``) and p = θ + 1 for A14. ``_k_h``'s ``1 − h``
member uses the cancellation-free form of this product
(``_nelsen_logh_nocancel``), and ``_k_inv_h`` solves ``h = w`` by a monotone
Newton iteration in ``D = log(t/a)`` (``_nelsen_inv_h_logb``) — closed-form
bounds, no bracketing solver, vectorised — as GH/Joe do and unlike BB1's
Brent fallback. The public ``inv_h``/``inv_h_array`` are still
:class:`CopulaVirt`'s (unchanged); ``_k_inv_h`` only serves the rotations.
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
# Log-space kernel on the pair ka = (log u, log(1 − u)) (module docstring) —
# shared by every evaluation path, including the rotation wrapper's kernel
# interface.
# ---------------------------------------------------------------------------

def _a12_logx(ka):
    """``log((1 − u)/u)`` from ``ka = (log u, log(1 − u))``."""
    return ka[1] - ka[0]


def _a12_terms(ka, kb, th):
    """``(log U1, log S, log t)`` with ``U1 = x^θ``, ``S = U1 + U2``, ``t = S^{1/θ}``."""
    log_u1 = th * _a12_logx(ka)
    log_u2 = th * _a12_logx(kb)
    log_s  = np.logaddexp(log_u1, log_u2)
    return log_u1, log_s, log_s / th


def _a12_logpdf(ka, kb, th):
    """log c(u, v) from ``ka = (log u, log(1 − u))``, ``kb = (log v, log(1 − v))``."""
    lu, l1u = ka
    lv, l1v = kb
    log_u1 = th * (l1u - lu)
    log_u2 = th * (l1v - lv)
    log_s  = np.logaddexp(log_u1, log_u2)
    log_t  = log_s / th
    with np.errstate(divide='ignore'):
        # (θ − 1) + (θ + 1) t: at θ = 1 the constant vanishes, and
        # logaddexp(−inf, x) = x handles that boundary exactly.
        log_last = np.logaddexp(np.log(th - 1.0), np.log(th + 1.0) + log_t)
    # log U_i − log u_i − log(1 − u_i) = (θ − 1) log(1 − u_i) − (θ + 1) log u_i
    return ((th - 1.0) * (l1u + l1v) - (th + 1.0) * (lu + lv)
            + (1.0 / th - 2.0) * log_s
            - 3.0 * np.logaddexp(0.0, log_t)
            + log_last)


def _a12_cdf(ka, kb, th):
    """``(C, 1 − C)``: C = 1/(1 + t) = exp(−log(1 + t)), 1 − C = t/(1 + t)."""
    _, _, log_t = _a12_terms(ka, kb, th)
    sp = np.logaddexp(0.0, log_t)
    return np.exp(-sp), np.exp(log_t - sp)


def _nelsen_logh_nocancel(la, lb, lr, th, p):
    """``log h`` in the cancellation-free form shared by A12 and A14.

    Both families have ``h(v|u) = (a/t)^{θ−1} ((1 + a)/(1 + t))^p`` with
    ``t = (a^θ + b^θ)^{1/θ}`` in their own transformed coordinates ``a``, ``b``
    of u, v (module docstrings; p = 2 for A12, θ + 1 for A14). With
    ``D = log(t/a) = logaddexp(0, θ(log b − log a))/θ ≥ 0`` and
    ``r = a/(1 + a) ∈ (0, 1)``::

        log h = −(θ − 1)·D − p·log1p(expm1(D)·r)

    both terms ≤ 0, so ``1 − h = −expm1(log h)`` keeps its relative precision
    where h → 1 (the textbook product, whose terms cancel there, does not).
    """
    d = np.logaddexp(0.0, th * (lb - la)) / th
    return -(th - 1.0) * d - p * np.log1p(np.expm1(d) * np.exp(lr))


def _nelsen_inv_h_logb(lw, la, lr, th, p, max_iter=100):
    """``log b`` solving h(v|u) = w for the shared A12/A14 form above.

    ``G(D) = (θ − 1)·D + p·log1p(expm1(D)·r) − q = 0``, q = −log w > 0, is
    increasing and convex in D ≥ 0 (``log(1 − r + r e^D)`` has second
    derivative ``r(1 − r)e^D/(1 − r + r e^D)² ≥ 0``), so Newton started at an
    upper bound of the root decreases monotonically to it — the monotone
    Newton iteration of ``gumbel.py``'s inverse h-function (vinecopulib's
    transformed-variable scheme). Upper bounds, each dropping one
    non-negative term of G: ``q/(θ − 1)`` (θ > 1) and
    ``log1p(expm1(q/p)/r)``; the start is the smaller. The iteration stops
    on a relative step of 4ε or a residual at G's rounding floor (16ε·q).
    At θ = 1 the first term vanishes and the second bound is the root
    itself. Then ``b^θ = t^θ − a^θ``: ``log b = log a + log(expm1(θD))/θ``.
    """
    q = -np.asarray(lw, dtype=float)
    la = np.asarray(la, dtype=float)
    lr = np.asarray(lr, dtype=float)
    r = np.exp(lr)
    tm1 = th - 1.0
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        # log1p(expm1(q/p)/r) = softplus(log expm1(q/p) − log r), no overflow
        e = q / p
        log_em1 = np.where(e > 1.0, e + np.log(-np.expm1(-e)), np.log(np.expm1(e)))
        d = np.logaddexp(0.0, log_em1 - lr)
        if tm1 > 0.0:
            d = np.minimum(d, q / tm1)
    done = np.zeros(np.shape(d), dtype=bool)
    for _ in range(max_iter):
        with np.errstate(over='ignore', invalid='ignore'):
            em1 = np.expm1(d)
            g = tm1 * d + p * np.log1p(em1 * r) - q
            # d/dD log1p(expm1(D) r) = r e^D / (1 + expm1(D) r)
            gp = tm1 + p * r * np.exp(d) / (1.0 + em1 * r)
            step = g / gp
        step = np.where(np.isfinite(step), step, 0.0)
        new = np.maximum(d - step, 0.0)
        # converged entries are frozen so that each result is independent of the batch
        done_now = ((np.abs(new - d) <= 4.0 * EPS * new) | (np.abs(g) <= 16.0 * EPS * q)
                    | ~np.isfinite(g))
        d = np.where(done, d, new)
        done |= done_now
        if np.all(done):
            break
    z = th * d
    with np.errstate(divide='ignore'):
        # log(expm1(z)) without overflow
        return la + (z + np.log(-np.expm1(-z))) / th


def _a12_h(kb, ka, th):
    """``(h, 1 − h)`` of h(v|u) = x^{θ−1} S^{1/θ−1} (1 + t)^{−2} / u².

    h is the textbook log-space product (unchanged, so the public
    :meth:`CopulaA12.conditional_cdf` keeps its values bit for bit);
    1 − h comes from the cancellation-free :func:`_nelsen_logh_nocancel`
    with ``a = x``, ``r = x/(1 + x) = 1 − u``, p = 2.
    """
    log_u1, log_s, log_t = _a12_terms(ka, kb, th)
    logh = (log_u1 - ka[0] - ka[1]
            + (1.0 / th - 1.0) * log_s
            - 2.0 * np.logaddexp(0.0, log_t))
    logh_nc = _nelsen_logh_nocancel(_a12_logx(ka), _a12_logx(kb), ka[1], th, 2.0)
    return np.exp(logh), -np.expm1(logh_nc)


def _a12_inv_h(lw, ka, th):
    """``(v, 1 − v)`` solving h(v|u) = w, from ``lw = log w`` and ``ka``.

    Monotone Newton (:func:`_nelsen_inv_h_logb`) for ``log y``,
    y = (1 − v)/v; then ``v = 1/(1 + y)`` and ``1 − v = y/(1 + y)``, each
    from ``log y`` without forming the complement.
    """
    ly = _nelsen_inv_h_logb(lw, _a12_logx(ka), ka[1], th, 2.0)
    sp = np.logaddexp(0.0, ly)
    return np.exp(-sp), np.exp(ly - sp)


class CopulaA12(CopulaVirt):

    def __init__(self, **kwargs):
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    def _update_params(self):
        self.theta = 2.0 / (3.0 * (1.0 - self.params['tau_k']))

    # ------------------------------------------------------------------
    # Kernel interface (audit FR-8) — consumed by the 90°/270° rotation
    # wrapper in ``pmcprg.copulas.archimedean.rotated``, same members as
    # ``CopulaGH``/``CopulaJoe``/``CopulaBB1``'s. The kernel coordinate of a
    # point x is the *pair* ``(log x, log(1 − x))`` (module docstring);
    # ``_kcoord_reflected`` swaps it, which is exact.
    # ------------------------------------------------------------------

    @staticmethod
    def _kcoord(x):
        return (np.log(x), np.log1p(-x))

    @staticmethod
    def _kcoord_reflected(x):
        return (np.log1p(-x), np.log(x))

    def _k_logpdf(self, ka, kb):
        return _a12_logpdf(ka, kb, self.theta)

    def _k_cdf(self, ka, kb):
        return _a12_cdf(ka, kb, self.theta)

    def _k_h(self, kb, ka):
        return _a12_h(kb, ka, self.theta)

    def _k_inv_h(self, lw, ka):
        return _a12_inv_h(lw, ka, self.theta)

    # ------------------------------------------------------------------
    # Public API — built from the kernel interface above
    # ------------------------------------------------------------------

    def pdf(self, uv):
        """c(u, v) = exp(log c), see the module docstring; 0.0 only on underflow."""
        u = minmaxEPS(uv[0])
        v = minmaxEPS(uv[1])
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            return float(np.exp(self._k_logpdf(self._kcoord(np.float64(u)),
                                               self._kcoord(np.float64(v)))))

    def cdf(self, uv):
        """C(u, v) = 1/(1 + S^{1/θ})."""
        u = minmaxEPS(uv[0])
        v = minmaxEPS(uv[1])
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            c, _ = self._k_cdf(self._kcoord(np.float64(u)), self._kcoord(np.float64(v)))
            return float(np.clip(c, 0.0, 1.0))

    def cdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised closed-form CDF (A12 has no statsmodels backend)."""
        uv = np.asarray(uv, dtype=float)
        u  = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v  = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            c, _ = self._k_cdf(self._kcoord(u), self._kcoord(v))
            return np.clip(c, 0.0, 1.0)

    def pdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised closed-form PDF, ``exp`` of :meth:`logpdf_array` (no floor)."""
        return np.exp(self.logpdf_array(uv))

    def logpdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Native log-PDF for A12, computed entirely in log space.

        With x = (1 − u)/u, y = (1 − v)/v, S = x^θ + y^θ, t = S^{1/θ}:

            log c = θ log x − log u − log(1 − u) + θ log y − log v − log(1 − v)
                  + (1/θ − 2) log S − 3 log(1 + t) + log((θ − 1) + (θ + 1) t).
        """
        uv = np.asarray(uv, dtype=float)
        u  = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v  = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            return self._k_logpdf(self._kcoord(u), self._kcoord(v))

    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = (1+S^{1/θ})^{−2} · S^{1/θ−1} · (1/u−1)^{θ−1} / u², in log space.

        The former linear-scale form overflowed as τ → 1 for v ≪ u (the brentq
        bracket of :meth:`inv_h`) and returned the limiting step 𝟙{v ≥ u}; the
        log-space kernel stays finite at any θ, so no fallback is needed.
        """
        u = minmaxEPS(u)
        v = minmaxEPS(v)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            h, _ = self._k_h(self._kcoord(np.float64(v)), self._kcoord(np.float64(u)))
            return float(np.clip(h, 0.0, 1.0))

    def tail_dependence(self) -> tuple[float, float]:
        """A12: λ_L = 2^{−1/θ}, λ_U = 2 − 2^{1/θ} (Nelsen 2006, family 4.2.12).

        From the diagonal C(u,u) = 1/(1 + 2^{1/θ}(1 − u)/u): C(u,u)/u → 2^{−1/θ}
        as u → 0, and (1 − 2u + C(u,u))/(1 − u) → 2 − 2^{1/θ} as u → 1. The
        inherited numeric limit at u = 10⁻⁴ was off by ~10⁻⁴ (audit RB-10).
        """
        return (float(2.0 ** (-1.0 / self.theta)),
                float(2.0 - 2.0 ** (1.0 / self.theta)))

    # Numerical majorant and the public inv_h/inv_h_array (Brent on
    # conditional_cdf) are inherited from CopulaVirt, unchanged by the
    # kernel refactor; ``_k_inv_h`` serves the rotations only.


if __name__ == '__main__':
    from pathlib import Path

    cop = CopulaA12(tau_k=0.5)
    print(f'Copula   : {cop.copula_enum.value.LONG_NAME}')
    print(f'tau_k    : {cop.params["tau_k"]:.4f}  range={cop.tau_range}')
    print(f'theta    : {cop.theta:.6f}  [= 2/(3(1−τ))]')
    print(f'pdf(0.3, 0.7) = {cop.pdf([0.3, 0.7]):.6f}')
    print(f'cdf(0.3, 0.7) = {cop.cdf([0.3, 0.7]):.6f}')
    print(f'h(0.7 | 0.3)  = {cop.conditional_cdf(0.7, 0.3):.6f}')
    lL, lU = cop.tail_dependence()
    print(f'tail dep : λ_L = {lL:.4f}  [= 2^(−1/θ)],  λ_U = {lU:.4f}  [= 2 − 2^(1/θ)]')

    plot_dir = Path('./data/Plots/Copulas')
    plot_dir.mkdir(parents=True, exist_ok=True)
    cop.plot_pdf(plot_dir)
    cop.plot_cdf(plot_dir)
    cop.plot_h_function(plot_dir)
    cop.plot_samples(plot_dir)
    cop.plot_overview(plot_dir)
    cop.plot_multi_tau(plot_dir)
