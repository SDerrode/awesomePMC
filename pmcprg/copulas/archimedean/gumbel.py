"""
Gumbel–Hougaard copula — positive upper-tail dependence.

Generator: φ(t) = (−ln t)^θ,  θ = 1/(1 − τ_K) ≥ 1.
CDF:  C(u,v) = exp(−A),  A = ((−ln u)^θ + (−ln v)^θ)^{1/θ}
PDF:  c(u,v) = C(u,v) · (ln u · ln v)^{θ−1} · A^{1−2θ} · (A + θ − 1) / (u v)
h:    h(v|u) = C(u,v) · (−ln u)^{θ−1} · A^{1−θ} / u
τ = 1 − 1/θ;  λ_L = 0,  λ_U = 2 − 2^{1/θ}.

Reference: Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed.,
Springer, Table 4.1 family (4.2.4); Joe, H. (1997). *Multivariate Models
and Dependence Concepts*, Chapman & Hall, ch. 5 (one-parameter families).

Numerics
--------
Evaluated in log space: ``log A = (1/θ)·logaddexp(θ log x̃, θ log ỹ)`` with
``x̃ = −ln u``, ``ỹ = −ln v``. Until audit K-1 this module delegated the
density to statsmodels and inherited the generic ``log(max(pdf, EPS))``,
which floors log c at −36 nat: for τ = 0.95 that floor covered a sixth of
the unit square, where the true log-density runs from −40 to −130 nat.

*Kernel coordinates* (audit FR-1). Every kernel takes ``ka = log a``, never
``a``: x̃ = −ka is then exact even when the survival copula (``survival.py``)
evaluates the base at 1 − u and passes ``log1p(−u)`` — forming 1 − u in
floating point put SGH's log-density off by up to 4·10⁻⁴ nat and its
h-function by 2.5·10⁻⁸ at u = 10⁻¹² (roadmap technique 8). With
``D = log A − log x̃ = logaddexp(0, θ(log ỹ − log x̃))/θ ≥ 0``::

    log c = −A + x̃ + ỹ + (θ − 1)(log x̃ + log ỹ) + (1 − 2θ) log A + log(A + θ − 1)
    C     = exp(−A),                     1 − C = −expm1(−A)
    log h = −x̃·expm1(D) − (θ − 1)·D     [h(b | a); both terms ≤ 0, 1 − h = −expm1(log h)]

the last line being log of the textbook h with ``A − x̃ = x̃·expm1(D)``
written without cancellation. At θ = 1, D = log1p(ỹ/x̃) and log h = −ỹ:
h(v|u) = v exactly.

*Inverse h-function* (no closed form). h(b | a) = w is
``G(D) = x̃·expm1(D) + (θ − 1)·D + log w = 0`` in the same D — Newton's
iteration in the transformed variable ``A = x̃ e^D`` of vinecopulib's
``gumbel.ipp`` (roadmap technique 16). G is increasing and convex, so Newton
started at the upper bound ``min(log1p(q/x̃), q/(θ − 1))`` (q = −log w, each
bound exact when its term dominates) decreases monotonically to the root —
no step halving, no absolute tolerance; it stops on a relative step of 4ε or
a residual at G's rounding floor (16ε·q). Then
``log ỹ = log x̃ + log(expm1(θD))/θ`` and ``b = exp(−ỹ)``, ``1 − b = −expm1(−ỹ)``.
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

def _gh_logpdf(ka, kb, th):
    xt = -ka
    yt = -kb
    log_xt = np.log(xt)
    log_yt = np.log(yt)
    log_a  = np.logaddexp(th * log_xt, th * log_yt) / th
    with np.errstate(divide='ignore'):
        # log(A + θ − 1): at θ = 1 the constant vanishes; logaddexp(x, −inf) = x.
        log_last = np.logaddexp(log_a, np.log(max(th - 1.0, 0.0)))
    return (-np.exp(log_a) + xt + yt
            + (th - 1.0) * (log_xt + log_yt)
            + (1.0 - 2.0 * th) * log_a
            + log_last)


def _gh_cdf(ka, kb, th):
    """``(C, 1 − C)``."""
    a = np.exp(np.logaddexp(th * np.log(-ka), th * np.log(-kb)) / th)
    return np.exp(-a), -np.expm1(-a)


def _gh_h(kb, ka, th):
    """``(h, 1 − h)`` of h(b | a)."""
    xt = -ka
    log_xt = np.log(xt)
    d = np.logaddexp(0.0, th * (np.log(-kb) - log_xt)) / th
    logh = -xt * np.expm1(d) - (th - 1.0) * d
    return np.exp(logh), -np.expm1(logh)


def _gh_inv_h(lw, ka, th, max_iter=100):
    """``(b, 1 − b)`` solving h(b | a) = w, from ``lw = log w`` and ``ka = log a``."""
    q = -np.asarray(lw, dtype=float)
    xt = -np.asarray(ka, dtype=float)
    tm1 = th - 1.0
    with np.errstate(divide='ignore'):
        d = np.minimum(np.log1p(q / xt), q / tm1 if tm1 > 0.0 else np.inf)
    done = np.zeros(np.shape(d), dtype=bool)
    for _ in range(max_iter):
        g = xt * np.expm1(d) + tm1 * d - q
        step = g / (xt * np.exp(d) + tm1)
        step = np.where(np.isfinite(step), step, 0.0)
        new = np.maximum(d - step, 0.0)
        # relative step, or residual at the rounding floor of G (≈ ε·q); converged
        # entries are frozen so that each result is independent of the batch.
        done_now = (np.abs(new - d) <= 4.0 * EPS * new) | (np.abs(g) <= 16.0 * EPS * q) | ~np.isfinite(g)
        d = np.where(done, d, new)
        done |= done_now
        if np.all(done):
            break
    z = th * d
    with np.errstate(divide='ignore'):
        log_yt = np.log(xt) + (z + np.log(-np.expm1(-z))) / th    # log(expm1(z)) without overflow
    yt = np.exp(log_yt)
    return np.exp(-yt), -np.expm1(-yt)


class CopulaGH(CopulaVirt):
    """
    Gumbel-Hougaard copula — positive upper-tail dependence.

    Generator: φ(t) = (−ln t)^θ,  θ = 1/(1 − τ_K).
    θ = 1 → independence;  θ → ∞ → comonotonicity.
    """

    def __init__(self, **kwargs):
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    def _update_params(self):
        self.theta = 1.0 / (1.0 - self.params['tau_k'])

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
        return _gh_logpdf(ka, kb, self.theta)

    def _k_cdf(self, ka, kb):
        return _gh_cdf(ka, kb, self.theta)

    def _k_h(self, kb, ka):
        return _gh_h(kb, ka, self.theta)

    def _k_inv_h(self, lw, ka):
        return _gh_inv_h(lw, ka, self.theta)

    # -- public API ----------------------------------------------------------

    def pdf(self, uv):
        u = np.float64(minmaxEPS(uv[0]))
        v = np.float64(minmaxEPS(uv[1]))
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            return float(np.exp(_gh_logpdf(np.log(u), np.log(v), self.theta)))

    def pdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised closed-form PDF, ``exp`` of :meth:`logpdf_array` (no floor)."""
        return np.exp(self.logpdf_array(uv))

    def logpdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Native log-PDF, evaluated in log space (module docstring)."""
        uv = np.asarray(uv, dtype=float)
        u  = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v  = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            return _gh_logpdf(np.log(u), np.log(v), self.theta)

    def cdf(self, uv):
        u = np.float64(minmaxEPS(uv[0]))
        v = np.float64(minmaxEPS(uv[1]))
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            c, _ = _gh_cdf(np.log(u), np.log(v), self.theta)
        return float(np.clip(c, 0.0, 1.0))

    def cdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised closed-form CDF (NaN where it cannot be evaluated)."""
        uv = np.asarray(uv, dtype=float)
        u  = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v  = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            c, _ = _gh_cdf(np.log(u), np.log(v), self.theta)
        return np.clip(c, 0.0, 1.0)

    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = C(u,v)/u · (−ln u)^{θ−1} · A^{1−θ}, in log space (module docstring)."""
        u = np.float64(minmaxEPS(u))
        v = np.float64(minmaxEPS(v))
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            h, _ = _gh_h(np.log(v), np.log(u), self.theta)
        return float(np.clip(h, 0.0, 1.0))

    def inv_h(self, w: float, u: float) -> float:
        """v such that h(v|u) = w, by the monotone Newton iteration of the module docstring."""
        return float(self.inv_h_array(np.array([w], dtype=float), np.array([u], dtype=float))[0])

    def inv_h_array(self, w: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Vectorised inverse h-function (monotone Newton, module docstring)."""
        w = np.clip(np.asarray(w, dtype=float), EPS, ONE_MINUS_EPS)
        u = np.clip(np.asarray(u, dtype=float), EPS, ONE_MINUS_EPS)
        if w.shape != u.shape:
            raise ValueError(f"inv_h_array: w {w.shape} and u {u.shape} must match.")
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            v, _ = _gh_inv_h(np.log(w), np.log(u), self.theta)
        return np.clip(v, EPS, ONE_MINUS_EPS)

    def tail_dependence(self) -> tuple[float, float]:
        """GH copula: λ_L = 0, λ_U = 2 − 2^{1/θ}."""
        return 0.0, 2.0 - 2.0 ** (1.0 / self.theta)

    # Numerical majorant (inherited from CopulaVirt)


if __name__ == '__main__':
    from pathlib import Path

    cop = CopulaGH(tau_k=0.5)
    lam_u = 2.0 - 2.0 ** (1.0 / cop.theta)
    print(f'Copula   : {cop.copula_enum.value.LONG_NAME}')
    print(f'tau_k    : {cop.params["tau_k"]:.4f}  range={cop.tau_range}')
    print(f'theta    : {cop.theta:.6f}  [= 1/(1−τ)]')
    print(f'pdf(0.3, 0.7) = {cop.pdf([0.3, 0.7]):.6f}')
    print(f'cdf(0.3, 0.7) = {cop.cdf([0.3, 0.7]):.6f}')
    print(f'h(0.7 | 0.3)  = {cop.conditional_cdf(0.7, 0.3):.6f}')
    print(f'tail dep : λ_L = 0,  λ_U = {lam_u:.4f}  [= 2 − 2^(1/θ)]')

    plot_dir = Path('./data/Plots/Copulas')
    plot_dir.mkdir(parents=True, exist_ok=True)
    cop.plot_pdf(plot_dir)
    cop.plot_cdf(plot_dir)
    cop.plot_h_function(plot_dir)
    cop.plot_samples(plot_dir)
    cop.plot_overview(plot_dir)
    cop.plot_multi_tau(plot_dir)
