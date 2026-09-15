"""
Gaussian copula — the copula of the bivariate normal distribution.

ρ = sin(π τ_K / 2)  (Kendall's τ ↔ correlation, as for every elliptical
copula);  ρ ∈ (−1, 1),  ρ = 0 → independence.
With x = Φ⁻¹(u), y = Φ⁻¹(v):
CDF:  C(u,v) = Φ_ρ(x, y)   (bivariate normal CDF — no closed form)
PDF:  log c(u,v) = −½ log(1 − ρ²) + (2ρxy − ρ²(x² + y²)) / (2(1 − ρ²))
h:    h(v|u) = Φ((y − ρx) / √(1 − ρ²)),  inverted in closed form by ``inv_h``
λ_L = λ_U = 0 for every |ρ| < 1 (no tail dependence).

Numerics
--------
*Quantiles.* x = Φ⁻¹(u) is ``ndtri(u)`` for u ≤ ½ and ``−ndtri(1 − u)``
above, so both tails keep relative precision (technique 11 of the numerics
notes); h uses ``ndtr``, whose lower tail is an ``erfc``.

*Log-density.* The quadratic form is written without cancellation near the
diagonal as ρ²(x² + y²) − 2ρxy = |ρ|·[|ρ|(x − σy)² − 2σ(1 − |ρ|)xy], σ = sign ρ,
and log(1 − ρ²) = log1p(−|ρ|) + log1p(|ρ|).

*CDF.* The former backend (statsmodels → scipy ``multivariate_normal.cdf``)
is a randomised quasi-Monte-Carlo integration with an absolute tolerance of
10⁻⁵: it returned 0 instead of 7.4·10⁻²⁰ at u = v = 10⁻¹⁴ and different
values on two calls. C is now the deterministic one-dimensional integral of
Drezner & Wesolowsky (1990), in the form of Genz (2004),

    Φ_ρ(x, y) = uv + (1/2π) ∫₀^{asin ρ} exp(−(x² + y² − 2xy sin t)/(2cos²t)) dt,

whose integrand is written without cancellation as
exp(−(x ∓ y)²/(2cos²t) ∓ xy/(1 ± sin t)) (upper signs for xy ≥ 0), with
1 ± sin t = 2 sin²(π/4 ± t/2). For ρ ≥ 0 both terms are positive, so the
relative precision is that of the quadrature. For ρ < 0 the integral is
subtracted from uv; where that loses more than two digits
(C < uv/100 — the lower-left tail) the same integrand is instead integrated
from the other Fréchet bound, Φ_ρ = max(0, u + v − 1) + (1/2π)∫_{−π/2}^{asin ρ},
again a sum of positive terms (max(0, u + v − 1) taken as
min(u,v) − (1 − max(u,v)), exact when max ≥ ½).

The quadrature is a composite 8-point Gauss–Legendre rule. Near t = ±π/2
the integrand behaves like exp(−(x ± y)²/(2 cos²t)), whose scale is the
distance to ±π/2, so the panels (width ≤ 0.3 rad) are halved toward asin ρ
down to min(0.02, 0.05·d), d = π/2 − |asin ρ|; the rule from −π/2 (panels
≤ min(0.2, d)) is halved toward asin ρ down to 10⁻³·d and toward −π/2 down
to 10⁻⁴·d — 10⁻¹²·d for the rare points with |x + y| < 10⁻³·d, whose layer
exp(−(x + y)²/(2cos²t)) is thinner. Nodes depend on ρ only: they are shared
by all points of an array and the evaluation is two outer products and an
``exp`` per node — 35 ms (τ = ½) to 125 ms (τ = −0.95) for 10⁵ points,
against ≈ 420 ms for the QMC backend. Against log-space adaptive-quadrature
references (the one of ``pmcprg/tests/test_palier1_b2_gaussian_cdf.py`` for
|ρ| ≤ 0.9999; for |ρ| closer to 1, the same probability integrated over Z in
Y = ρX + √(1−ρ²)Z), the relative error is below 2·10⁻¹⁰ for |ρ| ≤ 0.99 and
below 10⁻⁹ for 0.99 < |ρ| ≤ 1 − 10⁻¹², at u, v ∈ [10⁻¹⁴, 1 − 10⁻¹⁴] wherever
C > 10⁻³⁰⁰.

Reference: Aas, K., Czado, C., Frigessi, A. & Bakken, H. (2009).
Pair-copula constructions of multiple dependence. *Insurance: Mathematics
and Economics* 44(2), 182–198 (h-function); Demarta, S. & McNeil, A. J.
(2005). The t copula and related copulas. *International Statistical
Review* 73(1), 111–129 (elliptical copulas: τ = (2/π) arcsin ρ; the
Gaussian copula is the ν → ∞ limit of the t copula, with zero tail
dependence); Drezner, Z. & Wesolowsky, G. O. (1990). On the computation of
the bivariate normal integral. *Journal of Statistical Computation and
Simulation* 35, 101–107, doi:10.1080/00949659008811236; Genz, A. (2004).
Numerical computation of rectangular bivariate and trivariate normal and t
probabilities. *Statistics and Computing* 14, 251–260,
doi:10.1023/B:STCO.0000035304.20635.31.
"""
if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import math
import numpy as np
from scipy.special import ndtr, ndtri

from pmcprg.copulas._base   import CopulaVirt
from pmcprg.numerics     import EPS, MIN_POSITIVE, ONE_MINUS_EPS, EPS_MINUS_ONE, minmaxEPS


# ---------------------------------------------------------------------------
# Quantiles with both tails
# ---------------------------------------------------------------------------

def _norm_quantile(p):
    """Φ⁻¹(p): ``ndtri(p)`` for p ≤ ½, ``−ndtri(1 − p)`` above."""
    p = np.asarray(p, dtype=float)
    return np.where(p <= 0.5, ndtri(p), -ndtri(1.0 - p))


# ---------------------------------------------------------------------------
# Bivariate normal CDF (module docstring)
# ---------------------------------------------------------------------------

_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)
_PANEL_WIDTH = 0.3          # rad, rule from 0
_GRADE_ABS = 0.02           # rad, grading depth toward asin ρ (rule from 0) ...
_GRADE_REL = 0.05           # ... or this fraction of the distance from asin ρ to ±π/2
_TAIL_PANEL_WIDTH = 0.2     # rad, rule from −π/2 (at most the interval length)
_TAIL_GRADE_RHO = 1e-3      # grading toward asin ρ, relative to the interval length
_TAIL_GRADE_BOUND = 1e-4    # grading toward −π/2, relative ...
_TAIL_GRADE_THIN = 1e-12    # ... when |x + y| < _TAIL_THIN · length (thin layer at −π/2)
_TAIL_THIN = 1e-3
_TAIL_SWITCH = 1e2          # use the rule from −π/2 where C < uv / 100
_GRADE_FLOOR = 64.0 * EPS   # never grade below this distance (nodes stay off ±π/2)
_CHUNK = 4096               # points per outer-product block


def _theta_rule(a: float, b: float, width: float, grade_a, grade_b):
    """Composite 8-point Gauss–Legendre rule on [a, b], precomputed for the integrand.

    Panels of width ≤ ``width``; the end panels are halved repeatedly toward
    ``a`` (resp. ``b``) down to the distance ``grade_a`` (``grade_b``; None: no
    grading). Returns (1/cos²t, 1/(1 + sin t), 1/(1 − sin t), w/2π) at the nodes.
    """
    L = b - a
    if not L > 0.0:
        return None
    n = max(1, math.ceil(L / width))
    edges = list(np.linspace(a, b, n + 1))
    for depth, toward_a in ((grade_a, True), (grade_b, False)):
        if depth is None:
            continue
        d = 0.5 * L / n
        while d > max(depth, _GRADE_FLOOR):
            edges.append(a + d if toward_a else b - d)
            d *= 0.5
    edges = np.unique(edges)
    half = 0.5 * np.diff(edges)
    mid = 0.5 * (edges[1:] + edges[:-1])
    t = (mid[:, None] + half[:, None] * _GL_X).ravel()
    w = (half[:, None] * _GL_W).ravel()
    one_m_s = 2.0 * np.sin(np.pi / 4.0 - t / 2.0) ** 2
    one_p_s = 2.0 * np.sin(np.pi / 4.0 + t / 2.0) ** 2
    return 1.0 / (one_m_s * one_p_s), 1.0 / one_p_s, 1.0 / one_m_s, w / (2.0 * np.pi)


def _bvn_rules(rho: float) -> dict:
    """The quadrature rules for one ρ (module docstring)."""
    t_rho = math.asin(rho)
    if t_rho == 0.0:
        return {'from0': None}
    dist = math.pi / 2.0 - abs(t_rho)                 # from asin ρ to the singular ±π/2
    grade = min(_GRADE_ABS, _GRADE_REL * dist)
    if t_rho > 0.0:
        return {'from0': _theta_rule(0.0, t_rho, _PANEL_WIDTH, None, grade)}
    tail_width = min(_TAIL_PANEL_WIDTH, dist)
    return {
        'from0': _theta_rule(t_rho, 0.0, _PANEL_WIDTH, grade, None),
        'thin': _theta_rule(-math.pi / 2.0, t_rho, tail_width, _TAIL_GRADE_THIN * dist, _TAIL_GRADE_RHO * dist),
        'thick': _theta_rule(-math.pi / 2.0, t_rho, tail_width, _TAIL_GRADE_BOUND * dist, _TAIL_GRADE_RHO * dist),
        'dist': dist,
    }


def _theta_integral(x, y, rule):
    """(1/2π) ∫ exp(−(x² + y² − 2xy sin t)/(2 cos²t)) dt over the rule's interval."""
    inv_c2, inv_ps, inv_ms, w = rule
    out = np.empty(x.shape)
    for lo in range(0, x.size, _CHUNK):
        sl = slice(lo, lo + _CHUNK)
        xs, ys = x[sl], y[sl]
        xy = xs * ys
        pos = xy >= 0.0
        d = np.where(pos, xs - ys, xs + ys)
        E = np.multiply.outer(-0.5 * d * d, inv_c2)
        E -= np.abs(xy)[:, None] * np.where(pos[:, None], inv_ps, inv_ms)
        np.exp(E, out=E)
        E *= w
        out[sl] = E.sum(axis=1)                   # row-wise: independent of the batch
    return out


def _bvn_cdf(u, v, x, y, rho: float, rules: dict):
    """Φ_ρ(x, y) with Φ(x) = u, Φ(y) = v (module docstring)."""
    uv = u * v
    rule0 = rules['from0']
    if rule0 is None:
        return uv
    if rho > 0.0:
        return uv + _theta_integral(x, y, rule0)
    out = uv - _theta_integral(x, y, rule0)
    tail = out * _TAIL_SWITCH < uv
    if np.any(tail):
        # max(0, u + v − 1) as min − (1 − max): 1 − max is exact for max ≥ ½
        lower = np.maximum(0.0, np.minimum(u, v) - (1.0 - np.maximum(u, v)))
        thin = tail & (np.abs(x + y) < _TAIL_THIN * rules['dist'])
        for sel, rule in ((thin, rules['thin']), (tail & ~thin, rules['thick'])):
            if np.any(sel):
                out[sel] = lower[sel] + _theta_integral(x[sel], y[sel], rule)
    return out


class CopulaGaussian(CopulaVirt):

    def __init__(self, **kwargs):
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    def _update_params(self):
        self.theta = math.sin(self.params['tau_k'] * math.pi / 2.0)
        self.theta = max(EPS_MINUS_ONE, min(ONE_MINUS_EPS, self.theta))
        self._bvn_rules = None      # built on the first CDF evaluation

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @property
    def _one_minus_rho2(self) -> float:
        r = abs(self.theta)
        return (1.0 - r) * (1.0 + r)

    def _rules(self) -> dict:
        if self._bvn_rules is None:
            self._bvn_rules = _bvn_rules(self.theta)
        return self._bvn_rules

    # ------------------------------------------------------------------
    # PDF / CDF / h-function
    # ------------------------------------------------------------------
    def pdf(self, uv):
        """``exp`` of the native log-density, so the scalar and array paths agree (no floor)."""
        u = minmaxEPS(uv[0])
        v = minmaxEPS(uv[1])
        return float(np.exp(self.logpdf_array(np.array([[u, v]]))[0]))

    def pdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised PDF, ``exp`` of :meth:`logpdf_array` (no floor)."""
        with np.errstate(under='ignore', over='ignore'):
            return np.exp(self.logpdf_array(uv))

    def cdf(self, uv):
        """Deterministic bivariate normal CDF (module docstring)."""
        return float(self.cdf_array(np.array([[uv[0], uv[1]]], dtype=float))[0])

    def cdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised deterministic CDF — Drezner–Wesolowsky/Genz integral (module docstring)."""
        uv = np.asarray(uv, dtype=float)
        u = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        x, y = _norm_quantile(u), _norm_quantile(v)
        with np.errstate(under='ignore'):
            return np.clip(_bvn_cdf(u, v, x, y, self.theta, self._rules()), 0.0, 1.0)

    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = Φ((Φ⁻¹(v) − θ·Φ⁻¹(u)) / √(1−θ²)), quantiles with both tails."""
        x_u = _norm_quantile(minmaxEPS(u))
        x_v = _norm_quantile(minmaxEPS(v))
        return float(ndtr((x_v - self.theta * x_u) / math.sqrt(self._one_minus_rho2)))

    def inv_h(self, w: float, u: float) -> float:
        """Closed-form inverse of h(v|u): solve h(v|u) = w in one step.

        Derivation:  Φ⁻¹(w) = (Φ⁻¹(v) − θ·Φ⁻¹(u)) / √(1−θ²)
                  ⟹  v = Φ(Φ⁻¹(w)·√(1−θ²) + θ·Φ⁻¹(u))
        """
        return float(self.inv_h_array(np.array([w]), np.array([u]))[0])

    def inv_h_array(self, w: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Vectorised closed-form inverse h-function (M points at once).

        w is clipped to [MIN_POSITIVE, 1 − EPS] only, so an h-value below EPS
        still inverts to its own v; u and the result to [EPS, 1 − EPS].
        """
        w = np.clip(np.asarray(w, dtype=float), MIN_POSITIVE, ONE_MINUS_EPS)
        u = np.clip(np.asarray(u, dtype=float), EPS, ONE_MINUS_EPS)
        x_v = _norm_quantile(w) * math.sqrt(self._one_minus_rho2) + self.theta * _norm_quantile(u)
        return np.clip(ndtr(x_v), EPS, ONE_MINUS_EPS)

    def tail_dependence(self) -> tuple[float, float]:
        """Gaussian copula has zero tail dependence for any |θ| < 1."""
        return 0.0, 0.0

    def logpdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Native log-PDF, no floor (module docstring).

        For x = Φ⁻¹(u), y = Φ⁻¹(v), ρ = θ, σ = sign ρ:
            log c(u, v) = −½[log1p(−|ρ|) + log1p(|ρ|)]
                          − |ρ|[|ρ|(x − σy)² − 2σ(1 − |ρ|)xy] / (2(1 − ρ²))
        """
        uv = np.asarray(uv, dtype=float)
        u = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        x = _norm_quantile(u)
        y = _norm_quantile(v)
        r = abs(self.theta)
        sg = 1.0 if self.theta >= 0.0 else -1.0
        quad = r * (r * (x - sg * y) ** 2 - 2.0 * sg * (1.0 - r) * x * y)
        return (-0.5 * (math.log1p(-r) + math.log1p(r))
                - quad / (2.0 * self._one_minus_rho2))

    def majorant(self, u_left: float) -> float:
        """Closed-form maximum of c(u, v) over v.

        Derivation
        ----------
        Let x = Φ⁻¹(u), y = Φ⁻¹(v), ρ = θ. Then
            log c(u,v) = -½ log(1-ρ²) + (2ρxy - ρ²(x²+y²)) / (2(1-ρ²))
        ∂/∂y log c = -(ρ²y - ρx) / (1-ρ²) = 0  ⟹  y* = x/ρ.
        Plugging y* back into the exponent yields x²/2, so

            max_v c(u, v) = (1 - ρ²)^{-1/2} · exp((Φ⁻¹(u))² / 2).

        For ρ = 0 the copula PDF is identically 1, so the max is 1.
        """
        if abs(self.theta) < 1e-12:
            return 1.0
        x = float(_norm_quantile(minmaxEPS(u_left)))
        return float(math.exp(0.5 * x * x) / math.sqrt(self._one_minus_rho2))


if __name__ == '__main__':
    from pathlib import Path

    cop = CopulaGaussian(tau_k=0.5)
    print(f'Copula   : {cop.copula_enum.value.LONG_NAME}')
    print(f'tau_k    : {cop.params["tau_k"]:.4f}  range={cop.tau_range}')
    print(f'theta    : {cop.theta:.6f}  [= sin(π·τ/2)]')
    print(f'pdf(0.3, 0.7) = {cop.pdf([0.3, 0.7]):.6f}')
    print(f'cdf(0.3, 0.7) = {cop.cdf([0.3, 0.7]):.6f}')
    print(f'h(0.7 | 0.3)  = {cop.conditional_cdf(0.7, 0.3):.6f}')
    print('tail dep : λ_L = λ_U = 0  (for all |θ| < 1)')

    plot_dir = Path('./data/Plots/Copulas')
    plot_dir.mkdir(parents=True, exist_ok=True)
    cop.plot_pdf(plot_dir)
    cop.plot_cdf(plot_dir)
    cop.plot_h_function(plot_dir)
    cop.plot_samples(plot_dir)
    cop.plot_overview(plot_dir)
    cop.plot_multi_tau(plot_dir)
