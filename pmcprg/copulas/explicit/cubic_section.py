"""
Cubic-section copula — the symmetric copula with cubic sections of Nelsen
(2006), ch. 3 (copulas with cubic sections), with (A₁, A₂) = (B₂, B₁) = (4θ, 2θ).

CDF:  C(u,v) = uv[1 + 2θ(1−u)(1−v)(1 + u + v − 2uv)]
PDF:  c(u,v) = ∂²C/∂u∂v = 1 + (θ/4)(12st − 9s²t² + 3s² + 3t² − 1),
               s = 1 − 2u,  t = 1 − 2v   (see ``_cubsec_z``)
h:    h(v|u) = v[1 + 2θ(1−v)(1 + v − 3u² − 6uv + 6u²v)]
τ = 2θ/3 − 2θ²/75  (verified by quadrature to 1e-6), inverted in closed
form by ``_update_params``;  λ_L = λ_U = 0.

Admissible θ: at the corners c(0,0) = c(1,1) = 1 + 2θ and
c(1,0) = c(0,1) = 1 − 4θ, and the density is non-negative on the whole
square iff θ ∈ [−1/2, 1/4]. Positivity at (1,0) gives θ ≤ 1/4, hence
τ_max = 33/200. The registry keeps θ ∈ [0, 1/4], i.e. τ_K ∈ [0, 33/200];
θ ∈ [−1/2, 0) would be admissible too (τ down to −0.34) but is excluded
there.

Analytical majorant (verified, piecewise):
  u_left ∈ [0, 0.5] : M = 1 + 2θ(1 − 3·u²)
  u_left ∈ (0.5, 1] : M = 1 + 2θ(−3·u² + 6·u − 2)

Reference: Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed.,
Springer, ch. 3 (copulas with cubic sections); Nelsen, R. B., Quesada-Molina, J. J. &
Rodríguez-Lallena, J. A. (1997). Bivariate copulas with cubic sections.
*Journal of Nonparametric Statistics* 7(3), 205–220.

Numerics
--------
*θ(τ).* The root θ = (75/4)(2/3 − √(4/9 − 8τ/75)) cancels as τ → 0 (relative
error 8·10⁻⁴ at τ = 10⁻¹²); it is written θ = 2τ / (2/3 + √((100 − 24τ)/225))
instead, exact at τ = 0. At τ = 33/200 that rounds to 1/4 + 1 ulp, which
would make c(1, 0) = 1 − 4θ negative; θ is capped at 1/4, its exact value.

*Density.* The expanded polynomial is replaced by the (s, t) form above,
checked against it and against the mixed derivative of C to 50 digits.
For θ ≤ 1/8 the density is ≥ 1/2 and needs nothing more. For θ > 1/8 it
approaches 1 − 4θ, i.e. 0 at θ = 1/4, near the corners (0, 1) and (1, 0),
where 1 + (θ/4)E cancels. There (st < 0), with x = 2 min(u, 1 − u),
y = 2 min(v, 1 − v) (exact) and σ = x + y − xy,

    c = (1 − 4θ) + (θ/4) [24(x + y) − 30xy + 3(x² + y²) − 9σ²],

whose bracket is ≥ 0 and ≈ 24(x + y) near the corner — no cancellation. The
former ``np.maximum(c, EPS)`` floor is gone: at θ = 1/4 the density is
legitimately 0 at those corners (the kernel returns 0, hence log c = −inf,
at u, v ∈ {0, 1}) and 1.3·10⁻¹⁵ at the clipped corner (EPS, 1 − EPS), which
is what ``pdf_array`` now returns — ``exp(logpdf_array)`` to rounding
(audit RB-9).
"""
if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import math
import numpy as np

from pmcprg.copulas._base import CopulaVirt
from pmcprg.numerics   import EPS, ONE_MINUS_EPS, minmaxEPS


def _cubsec_z(u, v, th):
    """z = (θ/4)(12st − 9s²t² + 3s² + 3t² − 1), so c = 1 + z; 1-D arrays."""
    s = 1.0 - 2.0 * u
    t = 1.0 - 2.0 * v
    st = s * t
    return 0.25 * th * (12.0 * st - 9.0 * st * st + 3.0 * (s * s + t * t) - 1.0)


def _cubsec_near_zero(u, v, th):
    """c = (1 − 4θ) + (θ/4)[24(x+y) − 30xy + 3(x²+y²) − 9σ²], valid where st < 0."""
    x = 2.0 * np.minimum(u, 1.0 - u)
    y = 2.0 * np.minimum(v, 1.0 - v)
    sig = x + y - x * y
    return (1.0 - 4.0 * th) + 0.25 * th * (
        24.0 * (x + y) - 30.0 * x * y + 3.0 * (x * x + y * y) - 9.0 * sig * sig)


def _cubsec_pdf(u, v, th):
    """c on 1-D arrays, cancellation-free (module docstring).

    c ≥ 1 − 4θ, so z ≥ −½ for θ ≤ 1/8 and ``1 + z`` loses at most one digit;
    z < −½ happens only for θ > 1/8 and only where st < 0 (for st ≥ 0,
    c ≥ 1 − θ/4), and there the corner form is used on those points only.
    """
    z = _cubsec_z(u, v, th)
    c = 1.0 + z
    if th > 0.125:
        m = z < -0.5
        if m.any():
            c[m] = _cubsec_near_zero(u[m], v[m], th)
    return c


def _cubsec_logpdf(u, v, th):
    """log c on 1-D arrays: ``log1p(z)``, or log of the corner form where z < −½."""
    z = _cubsec_z(u, v, th)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.log1p(z)
        if th > 0.125:
            m = z < -0.5
            if m.any():
                out[m] = np.log(_cubsec_near_zero(u[m], v[m], th))
    return out


class CopulaCubSec(CopulaVirt):

    def __init__(self, **kwargs):
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    def _update_params(self):
        tau = self.params['tau_k']
        # Root of 2θ²/75 − 2θ/3 + τ = 0 in [0, 1/4], without the cancellation
        # of 2/3 − √(…) as τ → 0; capped at 1/4 against the 1-ulp rounding
        # excess at τ = 33/200 (see the module docstring).
        theta = 2.0 * tau / (2.0 / 3.0 + math.sqrt((100.0 - 24.0 * tau) / 225.0))
        self.theta = min(theta, 0.25)

    def pdf(self, uv):
        u1 = minmaxEPS(uv[0])
        u2 = minmaxEPS(uv[1])
        return float(_cubsec_pdf(np.array([u1]), np.array([u2]), self.theta)[0])

    def pdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised closed-form PDF (CubSec has no statsmodels backend); no floor."""
        uv = np.asarray(uv, dtype=float)
        u1 = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        u2 = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        return _cubsec_pdf(u1, u2, self.theta)

    def logpdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Native log-PDF of the cubic-section density (−inf where it is 0)."""
        uv = np.asarray(uv, dtype=float)
        u1 = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        u2 = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        return _cubsec_logpdf(u1, u2, self.theta)

    def cdf(self, uv):
        u1 = minmaxEPS(uv[0])
        u2 = minmaxEPS(uv[1])
        return u1 * u2 * (1.0 + 2.0*self.theta * (1.0 - u1) * (1.0 - u2) * (1.0 + u1 + u2 - 2.0*u1*u2))

    def cdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised closed-form CDF (CubSec has no statsmodels backend)."""
        uv = np.asarray(uv, dtype=float)
        u1 = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        u2 = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        result = u1 * u2 * (
            1.0 + 2.0 * self.theta * (1.0 - u1) * (1.0 - u2)
            * (1.0 + u1 + u2 - 2.0 * u1 * u2)
        )
        return np.clip(result, 0.0, 1.0)

    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = v · [1 + 2θ(1−v)(1+v−3u²−6uv+6u²v)]."""
        u = minmaxEPS(u)
        v = minmaxEPS(v)
        result = v * (1.0 + 2.0*self.theta*(1.0-v)*(1.0+v - 3.0*u*u - 6.0*u*v + 6.0*u*u*v))
        return float(np.clip(result, 0.0, 1.0))

    def majorant(self, u_left: float) -> float:
        """Analytical majorant (verified)."""
        u = u_left
        if u <= 0.5:
            return 1.0 + 2.0 * self.theta * (-3.0*u*u + 1.0)
        else:
            return 1.0 + 2.0 * self.theta * (-3.0*u*u + 6.0*u - 2.0)

    def tail_dependence(self) -> tuple[float, float]:
        """Cubic-section copula has zero tail dependence."""
        return 0.0, 0.0


if __name__ == '__main__':
    from pathlib import Path

    cop = CopulaCubSec(tau_k=0.15)
    print(f'Copula   : {cop.copula_enum.value.LONG_NAME}')
    print(f'tau_k    : {cop.params["tau_k"]:.4f}  range={cop.tau_range}')
    print(f'theta    : {cop.theta:.6f}')
    print(f'pdf(0.3, 0.7)  = {cop.pdf([0.3, 0.7]):.6f}')
    print(f'cdf(0.3, 0.7)  = {cop.cdf([0.3, 0.7]):.6f}')
    print(f'h(0.7 | 0.3)   = {cop.conditional_cdf(0.7, 0.3):.6f}')
    print(f'majorant(0.2)  = {cop.majorant(0.2):.6f}  [piecewise analytical]')
    print(f'majorant(0.8)  = {cop.majorant(0.8):.6f}')
    print('tail dep : λ_L = λ_U = 0  (τ_max = 33/200 = 0.165)')

    plot_dir = Path('./data/Plots/Copulas')
    plot_dir.mkdir(parents=True, exist_ok=True)
    cop.plot_pdf(plot_dir)
    cop.plot_cdf(plot_dir)
    cop.plot_h_function(plot_dir)
    cop.plot_samples(plot_dir)
    cop.plot_overview(plot_dir)
    cop.plot_multi_tau(plot_dir)
