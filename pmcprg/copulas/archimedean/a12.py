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
# Log-space kernel shared by every evaluation path
# ---------------------------------------------------------------------------

def _a12_logx(u):
    """``log((1 − u)/u)``, accurate at both ends of (0, 1)."""
    return np.log1p(-u) - np.log(u)


def _a12_terms(u, v, th):
    """``(log U1, log S, log t)`` with ``U1 = x^θ``, ``S = U1 + U2``, ``t = S^{1/θ}``."""
    log_u1 = th * _a12_logx(u)
    log_u2 = th * _a12_logx(v)
    log_s  = np.logaddexp(log_u1, log_u2)
    return log_u1, log_s, log_s / th


def _a12_logpdf(u, v, th):
    """log c(u, v) on arrays (or numpy scalars) already clipped to (0, 1)."""
    lu, l1u = np.log(u), np.log1p(-u)
    lv, l1v = np.log(v), np.log1p(-v)
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


def _a12_cdf(u, v, th):
    """C = 1/(1 + t) = exp(−log(1 + t))."""
    _, _, log_t = _a12_terms(u, v, th)
    return np.exp(-np.logaddexp(0.0, log_t))


def _a12_h(v, u, th):
    """h(v|u) = x^{θ−1} S^{1/θ−1} (1 + t)^{−2} / u², in log space."""
    log_u1, log_s, log_t = _a12_terms(u, v, th)
    return np.exp(log_u1 - np.log(u) - np.log1p(-u)
                  + (1.0 / th - 1.0) * log_s
                  - 2.0 * np.logaddexp(0.0, log_t))


class CopulaA12(CopulaVirt):

    def __init__(self, **kwargs):
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    def _update_params(self):
        self.theta = 2.0 / (3.0 * (1.0 - self.params['tau_k']))

    def pdf(self, uv):
        """c(u, v) = exp(log c), see the module docstring; 0.0 only on underflow."""
        u = minmaxEPS(uv[0])
        v = minmaxEPS(uv[1])
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            return float(np.exp(_a12_logpdf(np.float64(u), np.float64(v), self.theta)))

    def cdf(self, uv):
        """C(u, v) = 1/(1 + S^{1/θ})."""
        u = minmaxEPS(uv[0])
        v = minmaxEPS(uv[1])
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            return float(np.clip(_a12_cdf(np.float64(u), np.float64(v), self.theta), 0.0, 1.0))

    def cdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised closed-form CDF (A12 has no statsmodels backend)."""
        uv = np.asarray(uv, dtype=float)
        u  = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v  = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            return np.clip(_a12_cdf(u, v, self.theta), 0.0, 1.0)

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
            return _a12_logpdf(u, v, self.theta)

    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = (1+S^{1/θ})^{−2} · S^{1/θ−1} · (1/u−1)^{θ−1} / u², in log space.

        The former linear-scale form overflowed as τ → 1 for v ≪ u (the brentq
        bracket of :meth:`inv_h`) and returned the limiting step 𝟙{v ≥ u}; the
        log-space kernel stays finite at any θ, so no fallback is needed.
        """
        u = minmaxEPS(u)
        v = minmaxEPS(v)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            return float(np.clip(_a12_h(np.float64(v), np.float64(u), self.theta), 0.0, 1.0))

    def tail_dependence(self) -> tuple[float, float]:
        """A12: λ_L = 2^{−1/θ}, λ_U = 2 − 2^{1/θ} (Nelsen 2006, family 4.2.12).

        From the diagonal C(u,u) = 1/(1 + 2^{1/θ}(1 − u)/u): C(u,u)/u → 2^{−1/θ}
        as u → 0, and (1 − 2u + C(u,u))/(1 − u) → 2 − 2^{1/θ} as u → 1. The
        inherited numeric limit at u = 10⁻⁴ was off by ~10⁻⁴ (audit RB-10).
        """
        return (float(2.0 ** (-1.0 / self.theta)),
                float(2.0 - 2.0 ** (1.0 / self.theta)))

    # Numerical majorant (inherited from CopulaVirt)


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
