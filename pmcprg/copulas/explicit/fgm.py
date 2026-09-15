"""
Farlie–Gumbel–Morgenstern (FGM) copula — the copula with quadratic sections.

CDF:  C(u,v) = uv[1 + θ(1−u)(1−v)],  θ ∈ [−1, 1]
PDF:  c(u,v) = 1 + θ(1−2u)(1−2v)
h:    h(v|u) = v[1 + θ(1−v)(1−2u)]
τ = 2θ/9, so θ = 9τ_K/2 and τ_K ∈ [−2/9, 2/9];  λ_L = λ_U = 0.

Reference: Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed.,
Springer, ch. 3 (quadratic sections) and ch. 5 (τ = 2θ/9).

Numerics
--------
c ≥ 1 − |θ| > 0 for |θ| < 1, so the density needs no floor; the former
``np.maximum(c, EPS)`` is gone and ``pdf_array`` equals ``exp(logpdf_array)``
(audit RB-9). At the registered bounds |θ| = 1 the density vanishes at two
corners, where ``1 + θst`` (s = 1 − 2u, t = 1 − 2v) cancels; there, with
m_u = min(u, 1 − u) (so 1 − |s| = 2 m_u exactly),

    c = (1 − |θ|) + |θ| (2 m_u + |s| · 2 m_v)        whenever θst < 0,

a sum of non-negative terms. log c uses ``log1p(θst)`` away from that corner,
which is exact to rounding as θ → 0 (independence).
"""
if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import numpy as np
from pmcprg.copulas._base import CopulaVirt
from pmcprg.numerics   import EPS, ONE_MINUS_EPS, minmaxEPS


def _fgm_near_zero(u, v, th):
    """c = (1 − |θ|) + |θ|(2 m_u + |1 − 2u| · 2 m_v), valid where θst < 0."""
    a = abs(th)
    mu = np.minimum(u, 1.0 - u)
    mv = np.minimum(v, 1.0 - v)
    return (1.0 - a) + a * (2.0 * mu + np.abs(1.0 - 2.0 * u) * 2.0 * mv)


def _fgm_z(u, v, th):
    """z = θ(1−2u)(1−2v), so c = 1 + z; 1-D arrays."""
    return th * (1.0 - 2.0 * u) * (1.0 - 2.0 * v)


def _fgm_pdf(u, v, th):
    """c on 1-D arrays, cancellation-free.

    ``1 + z`` loses at most one digit while z ≥ −½; below, which requires
    |θ| > ½, the complement form of the module docstring is used on those
    points only (a mask, so the common case pays nothing).
    """
    z = _fgm_z(u, v, th)
    c = 1.0 + z
    if abs(th) > 0.5:
        m = z < -0.5
        if m.any():
            c[m] = _fgm_near_zero(u[m], v[m], th)
    return c


def _fgm_logpdf(u, v, th):
    """log c on 1-D arrays: ``log1p(z)``, or log of the complement form where z < −½."""
    z = _fgm_z(u, v, th)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.log1p(z)
        if abs(th) > 0.5:
            m = z < -0.5
            if m.any():
                out[m] = np.log(_fgm_near_zero(u[m], v[m], th))
    return out


class CopulaFGM(CopulaVirt):
    """
    Farlie-Gumbel-Morgenstern copula.

    C(u,v) = u·v·[1 + θ(1−u)(1−v)],  θ = 9/2 · τ_K.
    c(u,v) = 1 + θ(1−2u)(1−2v).
    τ_K ∈ [−2/9, 2/9].

    Analytical majorant:  M(u_L) = 1 + |θ| · |1 − 2·u_L|
    """

    def __init__(self, **kwargs):
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    def _update_params(self):
        self.theta = 4.5 * self.params['tau_k']

    def pdf(self, uv):
        u0 = minmaxEPS(uv[0])
        u1 = minmaxEPS(uv[1])
        return float(_fgm_pdf(np.array([u0]), np.array([u1]), self.theta)[0])

    def pdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised closed-form PDF (FGM has no statsmodels backend); no floor."""
        uv = np.asarray(uv, dtype=float)
        u = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        return _fgm_pdf(u, v, self.theta)

    def logpdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Native log-PDF: log(1 + θ(1−2u)(1−2v)), see the module docstring."""
        uv = np.asarray(uv, dtype=float)
        u = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        return _fgm_logpdf(u, v, self.theta)

    def cdf(self, uv):
        u0 = minmaxEPS(uv[0])
        u1 = minmaxEPS(uv[1])
        return u0 * u1 * (1.0 + self.theta * (1.0 - u0) * (1.0 - u1))

    def cdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised closed-form CDF (FGM has no statsmodels backend)."""
        uv = np.asarray(uv, dtype=float)
        u  = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v  = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        return np.clip(u * v * (1.0 + self.theta * (1.0 - u) * (1.0 - v)),
                       0.0, 1.0)

    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = v · [1 + θ(1−v)(1−2u)]."""
        u = minmaxEPS(u)
        v = minmaxEPS(v)
        result = v * (1.0 + self.theta * (1.0 - v) * (1.0 - 2.0*u))
        return float(np.clip(result, 0.0, 1.0))

    def majorant(self, u_left: float) -> float:
        """Analytical majorant (verified)."""
        return 1.0 + abs(self.theta * (1.0 - 2.0 * u_left))

    def tail_dependence(self) -> tuple[float, float]:
        """FGM has zero tail dependence at both extremes."""
        return 0.0, 0.0


if __name__ == '__main__':
    from pathlib import Path

    cop = CopulaFGM(tau_k=0.15)
    print(f'Copula   : {cop.copula_enum.value.LONG_NAME}')
    print(f'tau_k    : {cop.params["tau_k"]:.4f}  range={cop.tau_range}')
    print(f'theta    : {cop.theta:.6f}  [= 9/2 · τ]')
    print(f'pdf(0.3, 0.7) = {cop.pdf([0.3, 0.7]):.6f}')
    print(f'cdf(0.3, 0.7) = {cop.cdf([0.3, 0.7]):.6f}')
    print(f'h(0.7 | 0.3)  = {cop.conditional_cdf(0.7, 0.3):.6f}')
    print(f'majorant(0.5) = {cop.majorant(0.5):.6f}  [= 1 + |θ||1−2u|]')
    print('tail dep : λ_L = λ_U = 0  (τ_max = 2/9 ≈ 0.222)')

    plot_dir = Path('./data/Plots/Copulas')
    plot_dir.mkdir(parents=True, exist_ok=True)
    cop.plot_pdf(plot_dir)
    cop.plot_cdf(plot_dir)
    cop.plot_h_function(plot_dir)
    cop.plot_samples(plot_dir)
    cop.plot_overview(plot_dir)
    cop.plot_multi_tau(plot_dir)
