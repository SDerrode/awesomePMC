"""
Tawn copulas, types 1 and 2 — the two two-parameter restrictions of Tawn's
(1988) asymmetric logistic extreme-value model (FR-9, round 3).

The full model
--------------
With ``w = −ln u``, ``z = −ln v``, the stable tail dependence function of the
asymmetric logistic model is (Tawn 1988; Joe 2014, ch. 4 and ch. 6)

    ℓ(w, z) = (1 − ψ_u)·w + (1 − ψ_v)·z + [(ψ_u w)^θ + (ψ_v z)^θ]^{1/θ},
    θ ≥ 1,  ψ_u, ψ_v ∈ [0, 1],       C(u, v) = exp(−ℓ(w, z)),

and its Pickands function, in this package's convention ``t = w/(w + z)``
(the u-share, as in ``galambos.py`` and ``husler_reiss.py``), is

    A(t) = ℓ(t, 1 − t) = (1 − ψ_u)·t + (1 − ψ_v)·(1 − t)
                         + [(ψ_u t)^θ + (ψ_v (1 − t))^θ]^{1/θ}.

The brief's form "A(t) = (1 − ψ₁)(1 − t) + (1 − ψ₂)t + [(ψ₁(1 − t))^θ +
(ψ₂ t)^θ]^{1/θ}" is this one with ψ₂ = ψ_u and ψ₁ = ψ_v: which argument
each ψ multiplies is *only* meaningful once t is tied to u or to v, and the
literature ties it both ways (see "Which ψ is fixed" below). This module
therefore never says ψ₁/ψ₂: it names each weight by the margin whose
``−ln`` it multiplies.

Verified, not transcribed (30–40-digit ``mpmath``, script used during
development): A(0) = A(1) = 1; max(t, 1 − t) ≤ A(t) ≤ 1 and A'' ≥ 0 at 1000
random (t, θ, ψ_u, ψ_v) — no violation. Both bounds also follow by hand: the
ℓ^θ-norm lies between the ℓ^∞- and the ℓ¹-norm, so
``max(ψ_u t, ψ_v(1−t)) ≤ [·]^{1/θ} ≤ ψ_u t + ψ_v (1 − t)``; the right
inequality gives A ≤ 1, the left one A ≥ t + (1 − ψ_v)(1 − t) ≥ t (and
symmetrically ≥ 1 − t). A is convex as a linear term plus a norm.

Limits: θ = 1 or ψ_u·ψ_v = 0 → **independence** (A ≡ 1 — at θ = 1 the norm
is ψ_u t + ψ_v(1 − t) exactly); ψ_u = ψ_v = 1 → **Gumbel–Hougaard** with
parameter θ; θ → ∞ → the **Marshall–Olkin** copula
``C = min(u^{1−ψ_u}·v, u·v^{1−ψ_v})`` (the norm tends to the max) — *not* the
comonotone copula unless ψ_u = ψ_v = 1.

The two two-parameter types
---------------------------
The package's parameter machinery takes τ plus at most one extra parameter
(``_fit._two_parameter_spec``, ``EXTRA_PARAM_BOUNDS``,
``CopulaVirt.constructible_params``), so, as VineCopula does (families 104
and 204), one weight is fixed at 1 and the other is the free asymmetry
parameter ``psi`` ∈ (0, 1]:

* **Tawn type 1** (``CopulaTawn1``): ψ_u = 1, ψ_v = ψ —
  ``ℓ = (1 − ψ) z + [w^θ + (ψ z)^θ]^{1/θ}``.
* **Tawn type 2** (``CopulaTawn2``): ψ_u = ψ, ψ_v = 1 —
  ``ℓ = (1 − ψ) w + [(ψ w)^θ + z^θ]^{1/θ}``.

Type 2 is the transpose of type 1 at the same (θ, ψ):
``C₂(u, v) = C₁(v, u)`` (swapping u and v swaps w and z), so the two share
τ(θ, ψ) and λ_U and differ only in the direction of the asymmetry.

*Which ψ is fixed — how this was established.* VineCopula's C sources (no
network access here, so this rests on recollection of the code, not on a
fresh reading) evaluate ``C(u, v) = (uv)^{A(t)}`` with
``t = ln v / ln(uv)`` — the **v**-share, the opposite of this module's t —
and ``A(t) = (1 − β)t + (1 − α)(1 − t) + [(α(1 − t))^θ + (βt)^θ]^{1/θ}``,
with type 1 setting α = 1 (β = ψ free) and type 2 setting β = 1 (α = ψ
free). Rewriting in (w, z): ``ℓ = (1 − β) z + (1 − α) w + [(α w)^θ +
(β z)^θ]^{1/θ}``, so α weights w (= u) and β weights z (= v): type 1 fixes
the u-weight and frees the v-weight, type 2 the reverse — the assignment
above. Because the two types are exact transposes, a reader whose source
uses the other convention only has to swap the two class names; nothing
else changes. The symmetric part of the claim (types 1 and 2 are the
ψ₂ = 1 / ψ₁ = 1 restrictions, and transposes of each other) was checked
numerically: τ agrees to 40 digits at (θ, ψ) = (3, 0.4), and the module
tests check C₂(u, v) = C₁(v, u) to rounding.

Kendall's τ and the reachable-τ cap
-----------------------------------
No closed form (as for Galambos and Hüsler–Reiss): τ(θ, ψ) is the Genest &
MacKay (1986) integral ``∫ t(1−t) A''(t)/A(t) dt``. It is computed by
:func:`pmcprg.copulas.extreme_value._pickands.tau_from_A_terms_gl`, a
vectorised composite Gauss–Legendre rule added to the shared module for this
family, on a mesh graded towards the ridge ``t* = ψ_v/(ψ_u + ψ_v)`` (the kink
of the θ → ∞ limit — not 1/2 unless ψ_u = ψ_v) at scale t*(1 − t*)/θ, and
towards both ends. Why not the Galambos/Hüsler–Reiss ``quad``: with its
breakpoints at 1/2 ± 5/θ (or even at t* ± 5·t*(1−t*)/θ) it is off by 4·10⁻⁹
relative at θ ≥ 10³ — Gumbel included — and it costs 3–7 ms per τ, where
the graded rule costs 0.2 ms; the joint fit and the multistart repair invert
τ(θ, ψ) far more often than a one-parameter family does.

Checked against a 60-digit ``mpmath`` quadrature whose A'' is a central
second difference of A itself (not this module's closed form), θ from
1 + 10⁻⁶ to 10⁶, ψ from 0.01 to 1, both types: relative error ≤ 3·10⁻¹³ for
θ ≤ 50 and ≤ 10⁻¹⁰ at θ = 10⁶ (the double-precision integrand's own limit
on a ridge of width 10⁻⁷); the rule reproduces Gumbel's τ = 1 − 1/θ at
ψ = 1. τ is increasing in θ at fixed ψ and in ψ at fixed θ on every grid
tried (θ ∈ [1 + 10⁻⁶, 10⁶], ψ ∈ [0.01, 1]; no proof attempted). At
ψ_u = ψ_v = 1 the model *is* Gumbel, and the module uses Gumbel's closed
forms τ = 1 − 1/θ, θ = 1/(1 − τ) there, so the ψ = 1 member builds the very
θ ``CopulaGH`` builds (a 10⁻¹³ quadrature error would move θ's last digits).

**The reachable τ is capped at ψ.** As θ → ∞, A tends to the Marshall–Olkin
Pickands function ``(1 − ψ_u)t + (1 − ψ_v)(1 − t) + max(ψ_u t, ψ_v(1 − t))``,
whose only curvature is a slope jump of ψ_u + ψ_v at t*. The Stieltjes form
of the Genest–MacKay identity then gives, by hand,

    τ_∞ = t*(1 − t*)·(ψ_u + ψ_v)/A(t*) = ψ_u ψ_v / (ψ_u + ψ_v − ψ_u ψ_v),

the known Marshall–Olkin value (Nelsen 2006, §5.1.1), which is **ψ** when
the other weight is 1: both types reach only τ < ψ. (``mpmath`` at
θ = 10⁵: τ = 0.3999984 for ψ = 0.4.) (τ, ψ) is therefore *jointly*
constrained, like BB1's (τ, δ): the constructor refuses τ ≥ ψ, and
:meth:`_CopulaTawn.constructible_params` repairs a refused pair (below).

θ is capped at ``_THETA_HI`` = 10⁶ (quadrature checked there, above); for
τ between τ(10⁶, ψ) ≈ ψ − O(10⁻⁶) and ψ the copula is built at the cap and
stores the τ it realises (RB-10, as Galambos). The lower end is θ = 1 + ε: τ ≈
c(ψ)·(θ − 1) with c(ψ) = ψ ln(1/ψ)/(1 − ψ) ∈ (0, 1], so τ(1 + ε, ψ) < ε ≤ any
registered τ and the low end never clamps. The inversion is Brent's method on
``log(θ − 1)`` — a Brent search on θ itself with an absolute tolerance of
10⁻¹² would leave θ − 1 ≈ 10⁻¹² (τ ≈ 10⁻¹²) with a relative error of order
one. Note that θ is a double: for τ ≲ 10⁻¹² the grid spacing of 1 + (θ − 1)
limits the relative accuracy of θ − 1, exactly as for Gumbel's θ = 1/(1 − τ).

Tail dependence
---------------
λ_L = 0 and, from λ_U = 2(1 − A(1/2)) (Nelsen 2006, Thm 5.7.3),

    λ_U = ψ_u + ψ_v − (ψ_u^θ + ψ_v^θ)^{1/θ} = ψ − ((1 + ψ^θ)^{1/θ} − 1)

for both types (2 − 2^{1/θ} at ψ = 1, Gumbel's value; → ψ as θ → ∞; → 0 as
θ → 1).

Densities (derivation)
----------------------
With N = [x^θ + y^θ]^{1/θ}, x = ψ_u w, y = ψ_v z, p = x^θ/N^θ, q = 1 − p,
differentiating ℓ directly (no transcription):

    ℓ_w  = (1 − ψ_u) + ψ_u p^{1 − 1/θ},     ℓ_z = (1 − ψ_v) + ψ_v q^{1 − 1/θ},
    ℓ_wz = −(θ − 1) ψ_u ψ_v (pq)^{1 − 1/θ} / N  (≤ 0),

and, from C = exp(−ℓ) and ∂/∂u = −(1/u)∂/∂w (the general EV identities
re-derived in ``husler_reiss.py``),

    h(v|u) = ∂C/∂u = C·ℓ_w/u,     c(u, v) = C·(ℓ_w ℓ_z − ℓ_wz)/(uv).

For the τ quadrature, Euler's relation for the degree-0 functions ℓ_w, ℓ_z
gives ``A''(t) = −ℓ_wz(t, 1 − t)/(t(1 − t))``.

Numerics
--------
Every term above is a sum of non-negative quantities — there is no
subtraction to cancel, unlike Galambos's ``1 − S·r/w``. What remains is to
avoid forming powers: with ``X = θ ln x``, ``Y = θ ln y``,

    ln N = logaddexp(X, Y)/θ,  ln p = −softplus(Y − X),  ln q = −softplus(X − Y),
    ln ℓ_w = logaddexp(ln(1 − ψ_u), ln ψ_u + (1 − 1/θ) ln p)     (idem ℓ_z),
    ln c = (ψ_u w + ψ_v z − N) + logaddexp(ln ℓ_w + ln ℓ_z,
                     ln(θ − 1) + ln ψ_u + ln ψ_v + (1 − 1/θ)(ln p + ln q) − ln N),
    ln h = (ψ_u w − N − (1 − ψ_v) z) + ln ℓ_w,       ln C = −ℓ.

``ψ_u w − N`` is formed as ``−x·expm1(softplus(Y − X)/θ)`` (Gumbel's
cancellation-free form; idem ``ψ_v z − N``), and ``x + y − N`` as
``(x − N) + y`` or ``x + (y − N)``, whichever bracket is the small one —
adding the larger of x, y to −N lost 5·10⁻¹⁴ nat at (1 − 10⁻⁶, 10⁻¹²). Being
additive terms of ln c and ln h, their remaining rounding is an absolute
error of a few 10⁻¹⁴ nat on the (10⁻¹², 1 − 10⁻¹²) grid. ln(1 − ψ) = −∞ at
ψ = 1 is absorbed exactly by logaddexp. No overflow at θ = 10⁶ or w, z down to
10⁻¹⁶: X, Y are plain products of logs.

``inv_h`` has no closed form. :meth:`_CopulaTawn.inv_h_array` solves
``ln h(v|u) = ln w`` by vectorised bisection on ``logit v`` over
``[logit ε, logit(1 − ε)]`` (68 halvings: bracket width < 10⁻¹⁹ in logit),
with the base class's saturation convention at the ends; the scalar
:meth:`inv_h` calls it on one point. It exists for speed — sampling 20 000
pairs through the base class's per-point Brent search takes minutes here.

Fitting
-------
``fit(method='mle')`` (the default) is the joint two-parameter MLE of
:func:`pmcprg.copulas._fit._fit_two_parameter_mle`, run for ``psi`` in its
own coordinates ``(ln(θ − 1), ψ)`` whose box *is* the admissible set (any
θ > 1, ψ ∈ (0, 1] is a Tawn copula) — see ``_two_parameter_spec``, which
runs this box twice (the second pass is a restart with a fresh curvature
memory: the likelihood is a narrow curved ridge in these coordinates, and
from a poor start a single L-BFGS-B pass stopped short on 3 of 50 contrived
starts, once by 131 nat; with the restart, 0 of 50). τ alone
cannot identify ψ, so ``fit(method='tau')`` logs a warning and falls back to
MLE (BB1's precedent).

References
----------
* Tawn, J. A. (1988). Bivariate extreme value theory: models and
  estimation. *Biometrika* 75(3), 397–415, doi:10.1093/biomet/75.3.397.
* Joe, H. (2014). *Dependence Modeling with Copulas*, Chapman & Hall/CRC,
  ch. 4 and ch. 6 (asymmetric Gumbel / Tawn model, EV-copula identities).
* Schepsmeier, U., Stoeber, J., Brechmann, E. C., Graeler, B., Nagler, T. &
  Erhardt, T. (2018). *VineCopula: Statistical Inference of Vine Copulas*,
  R package (families 104 and 204, Tawn types 1 and 2).
* Genest, C. & MacKay, J. (1986). The joy of copulas: bivariate
  distributions with uniform marginals. *The American Statistician* 40(4),
  280–283.
* Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed., Springer,
  §5.1.1 (Kendall's τ of the Marshall–Olkin family) and Thm 5.7.3.
"""
if __name__ == '__main__':
    import sys
    import pathlib

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import logging
import math
from collections import OrderedDict

import numpy as np
from scipy.optimize import brentq

from pmcprg.copulas._base import TAU_PAD_ABS, TAU_PAD_REL, CopulaVirt
from pmcprg.copulas.extreme_value._pickands import tau_from_A_terms_gl
from pmcprg.exceptions import CopulaParameterError
from pmcprg.numerics import EPS, ONE_MINUS_EPS, minmaxEPS

logger = logging.getLogger(__name__)

_THETA_LO = float(np.nextafter(1.0, 2.0))   # 1 + ε: τ(θ_LO, ψ) < ε for every ψ ≤ 1
_THETA_HI = 1.0e6                           # quadrature checked there (module docstring)
_PSI_DEFAULT = 1.0                          # a block without psi is the Gumbel member


# ---------------------------------------------------------------------------
# Log-space kernel (module docstring, "Numerics"). ``pu``, ``pv`` are the
# weights ψ_u, ψ_v of −ln u and −ln v.
# ---------------------------------------------------------------------------

def _log_weights(pu, pv):
    with np.errstate(divide='ignore'):
        return (math.log(pu), math.log(pv),
                math.log1p(-pu) if pu < 1.0 else -math.inf,
                math.log1p(-pv) if pv < 1.0 else -math.inf)


def _tawn_parts(w, z, theta, pu, pv):
    """Kernel terms at (w, z) > 0 (module docstring, "Numerics").

    Returns ``(xN, yN, N, log_lw, log_lz, log_mlwz)`` with ``xN = ψ_u w − N``
    and ``yN = ψ_v z − N`` (both ≤ 0), ``N`` the ℓ^θ-norm, and the logs of
    ℓ_w, ℓ_z and −ℓ_wz.
    """
    lpu, lpv, l1pu, l1pv = _log_weights(pu, pv)
    lx = lpu + np.log(w)
    ly = lpv + np.log(z)
    X = theta * lx
    Y = theta * ly
    sp_yx = np.logaddexp(0.0, Y - X)          # softplus(Y − X) = −ln p
    sp_xy = np.logaddexp(0.0, X - Y)          # softplus(X − Y) = −ln q
    # N = x·exp(softplus(Y − X)/θ) = y·exp(softplus(X − Y)/θ), so
    # x − N = −x·expm1(softplus(Y − X)/θ) without cancellation (Gumbel's form).
    xN = -np.exp(lx) * np.expm1(sp_yx / theta)
    yN = -np.exp(ly) * np.expm1(sp_xy / theta)
    logN = lx + sp_yx / theta
    e = 1.0 - 1.0 / theta
    log_lw = np.logaddexp(l1pu, lpu - e * sp_yx)
    log_lz = np.logaddexp(l1pv, lpv - e * sp_xy)
    with np.errstate(divide='ignore'):
        log_mlwz = math.log(theta - 1.0) + lpu + lpv - e * (sp_yx + sp_xy) - logN
    return xN, yN, np.exp(logN), log_lw, log_lz, log_mlwz


def _tawn_logpdf(ka, kb, theta, pu, pv):
    """ln c(u, v) from ``ka = ln u``, ``kb = ln v``.

    ``x + y − N`` is formed as ``(x − N) + y`` when x ≥ y and as ``x + (y − N)``
    otherwise: the bracket is then the small term, and adding the larger of
    x, y to ``−N`` directly would cost ≈ 5·10⁻¹⁴ nat at (u, v) = (1 − 10⁻⁶,
    10⁻¹²) — measured against ``mpmath``, where ``CopulaGH`` is 1·10⁻¹⁴ off.
    """
    w, z = -ka, -kb
    xN, yN, _, log_lw, log_lz, log_mlwz = _tawn_parts(w, z, theta, pu, pv)
    x_ge_y = np.log(pu) + np.log(w) >= np.log(pv) + np.log(z)
    with np.errstate(invalid='ignore'):
        excess = np.where(x_ge_y, xN + pv * z, yN + pu * w)
    return excess + np.logaddexp(log_lw + log_lz, log_mlwz)


def _tawn_log_cdf(ka, kb, theta, pu, pv):
    """ln C(u, v) = −ℓ(w, z) (≤ 0)."""
    w, z = -ka, -kb
    lpu, lpv, _, _ = _log_weights(pu, pv)
    X = theta * (lpu + np.log(w))
    Y = theta * (lpv + np.log(z))
    N = np.exp(np.logaddexp(X, Y) / theta)
    return -((1.0 - pu) * w + (1.0 - pv) * z + N)


def _tawn_logh(kb, ka, theta, pu, pv):
    """ln h(v | u) = ln ∂C(u, v)/∂u, from ``kb = ln v``, ``ka = ln u``."""
    w, z = -ka, -kb
    xN, _, _, log_lw, _, _ = _tawn_parts(w, z, theta, pu, pv)
    return (xN - (1.0 - pv) * z) + log_lw


# ---------------------------------------------------------------------------
# Pickands function and τ(θ)
# ---------------------------------------------------------------------------

def _tawn_A_terms(t, theta, pu, pv):
    """``(A, A', A'')`` at ``t`` (array or scalar), closed form (module docstring).

    A Gauss–Legendre node that rounds to t = 1 (1 − t = 0) gets A'' = 0, its
    true contribution t(1 − t)A'' = 0 to the τ integrand."""
    t = np.asarray(t, dtype=float)
    tm = 1.0 - t
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        _, _, N, log_lw, log_lz, log_mlwz = _tawn_parts(t, tm, theta, pu, pv)
    A = (1.0 - pu) * t + (1.0 - pv) * tm + N
    Ap = np.exp(log_lw) - np.exp(log_lz)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        App = np.exp(log_mlwz - np.log(t) - np.log(tm))
    App = np.where(np.isfinite(App), App, 0.0)
    return A, Ap, App


def _tawn_tau_quad(theta, pu, pv):
    """τ(θ; ψ_u, ψ_v) by the shared graded Gauss–Legendre Genest–MacKay rule,
    centred on the ridge t* = ψ_v/(ψ_u + ψ_v) with width t*(1 − t*)/θ
    (module docstring)."""
    theta = float(theta)
    if theta <= 1.0:
        return 0.0
    ts = pv / (pu + pv)
    return tau_from_A_terms_gl(lambda t, th: _tawn_A_terms(t, th, pu, pv), theta,
                               center=ts, width=ts * (1.0 - ts) / theta)


def _tawn_tau(theta, pu, pv):
    """τ(θ; ψ_u, ψ_v): Gumbel's closed form ``(θ − 1)/θ`` at ψ_u = ψ_v = 1 —
    the model *is* Gumbel there, and inverting the quadrature (≤ 10⁻¹¹ off,
    module tests) would build a θ that differs from ``CopulaGH``'s in its last
    digits — else the quadrature."""
    if pu == 1.0 and pv == 1.0:
        theta = float(theta)
        return 0.0 if theta <= 1.0 else (theta - 1.0) / theta
    return _tawn_tau_quad(theta, pu, pv)


# Memo of the forward map, and of its exact inverse on the values it produced:
# the joint MLE optimises in θ and hands the constructor τ(θ), which would
# otherwise re-invert τ by Brent's method (≈ 40 quadratures) at every
# likelihood evaluation. A hit returns the θ that produced τ — the exact
# inverse up to the quadrature's own accuracy, better than a Brent root. The
# price is a history dependence far below every tolerance: a τ rebuilt in a
# process that never produced it (or long after) goes through Brent (xtol
# 1e-13 on ln(θ − 1)) and may land ≈ 10⁻¹³ away. Never used at ψ = 1
# (Gumbel's closed form).
_MEMO_SIZE = 1 << 14
_TAU_MEMO: "OrderedDict[tuple, float]" = OrderedDict()
_THETA_MEMO: "OrderedDict[tuple, float]" = OrderedDict()


def _remember(memo, key, value):
    memo[key] = value
    memo.move_to_end(key)
    if len(memo) > _MEMO_SIZE:
        memo.popitem(last=False)


def _tau_of(theta, pu, pv):
    """τ(θ; ψ_u, ψ_v), memoised; records θ as the inverse of the returned τ on
    *every* call (hit or miss), so a constructor called right after with this
    τ gets this very θ back whatever the memo held before — the joint MLE's
    path does not depend on the call history."""
    key = (float(theta), float(pu), float(pv))
    tau = _TAU_MEMO.get(key)
    if tau is None:
        tau = _tawn_tau(*key)
        _remember(_TAU_MEMO, key, tau)
    if tau > 0.0:
        _remember(_THETA_MEMO, (tau, key[1], key[2]), key[0])
    return tau


def _theta_of(tau, pu, pv, family_name):
    """θ with τ(θ; ψ_u, ψ_v) = τ, Brent on ln(θ − 1); clamped at ``_THETA_HI``."""
    tau = float(tau)
    if tau <= 0.0:
        raise CopulaParameterError(
            f'{family_name}: tau_k={tau!r} must be > 0 (no negative or zero '
            f'dependence in an extreme-value copula).')
    if pu == 1.0 and pv == 1.0:
        # Gumbel: the same map as ``CopulaGH`` (θ = 1/(1 − τ)), capped —
        # before the memo, so the ψ = 1 member never depends on history.
        return min(1.0 / (1.0 - tau), _THETA_HI)
    hit = _THETA_MEMO.get((tau, float(pu), float(pv)))
    if hit is not None:
        return hit
    if tau >= _tau_of(_THETA_HI, pu, pv):
        return _THETA_HI
    s_lo, s_hi = math.log(_THETA_LO - 1.0), math.log(_THETA_HI - 1.0)

    def f(s):
        return _tau_of(1.0 + math.exp(s), pu, pv) - tau

    if f(s_lo) >= 0.0:
        return _THETA_LO
    s = brentq(f, s_lo, s_hi, xtol=1e-13, rtol=4.0 * np.finfo(float).eps, maxiter=200)
    theta = 1.0 + math.exp(s)
    _remember(_THETA_MEMO, (tau, float(pu), float(pv)), theta)
    return theta


# ---------------------------------------------------------------------------
# Copula classes
# ---------------------------------------------------------------------------

class _CopulaTawn(CopulaVirt):
    """Shared implementation of the Tawn types; see the module docstring.

    Parameters
    ----------
    tau_k : float
        Kendall's τ, 0 < τ < ψ (the reachable-τ cap, module docstring).
    psi : float, optional
        Asymmetry ψ ∈ (0, 1]; default 1.0 (the Gumbel–Hougaard member).
    """

    n_params: int = 2
    _FIXED_U: bool = True     # type 1: ψ_u = 1, ψ_v = ψ; type 2: the reverse

    def __init__(self, **kwargs):
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    # -- parameters ------------------------------------------------------

    @classmethod
    def _weights(cls, psi):
        return (1.0, psi) if cls._FIXED_U else (psi, 1.0)

    @classmethod
    def _psi_error(cls, tau, psi):
        """Why the constructor refuses (τ, ψ), or ``None`` when it accepts."""
        if not np.isfinite(psi) or not 0.0 < psi <= 1.0:
            return f'{cls.__name__}: psi must be in (0, 1], got {psi!r}.'
        if not tau < psi:
            return (f'{cls.__name__}: tau_k={tau!r} is not reachable with psi={psi!r} — '
                    f'the Tawn model needs tau < psi (its theta → ∞ limit, the '
                    f'Marshall–Olkin copula, has tau = psi).')
        return None

    @classmethod
    def reachable_tau_bounds(cls) -> tuple[float, float]:
        """``(EPS, τ(θ = 10⁶, ψ = 1))`` — the ψ = 1 (Gumbel) cap, the largest over ψ."""
        return float(EPS), _tau_of(_THETA_HI, 1.0, 1.0)

    @classmethod
    def repaired_psi(cls, tau: float) -> float:
        """The ψ :meth:`constructible_params` moves a refused pair to.

        The admissible ψ-interval at τ is (τ, 1]. Unlike BB1, whose refused
        end is the benign Gumbel limit, Tawn's refused end ψ → τ⁺ is the
        *singular* Marshall–Olkin limit θ → ∞: a start "just inside" it would
        have θ of order 10⁴ or beyond the cap. ψ goes instead to the middle
        of the interval, ``(1 + τ)/2``; if even that needs θ beyond
        ``_THETA_HI`` (τ within ≈ 10⁻⁶ of 1), to 1 — Gumbel, which builds at
        every τ the registry lets through.
        """
        tau = float(tau)
        psi = 0.5 * (1.0 + tau)
        pad = max(TAU_PAD_REL * (1.0 - tau), TAU_PAD_ABS)
        if psi - tau < pad or tau >= _tau_of(_THETA_HI, *cls._weights(psi)):
            return 1.0
        return psi

    @classmethod
    def constructible_params(cls, params: dict) -> dict:
        """``params`` itself when the constructor accepts it; else ψ moved (see
        :meth:`repaired_psi`), τ kept.

        The test is the constructor's own: ψ ∈ (0, 1] and τ < ψ. A pair the
        constructor accepts is returned untouched (same object), including
        one it builds at the θ cap. A missing ψ is the default 1.0, accepted
        at every τ < 1. A pair no ψ repairs (τ ≥ 1, τ ≤ 0, non-finite τ) is
        returned unchanged for the constructor to refuse.
        """
        tau = params["tau_k"]
        psi = params.get("psi", _PSI_DEFAULT)
        try:
            tau_f, psi_f = float(tau), float(psi)
        except (TypeError, ValueError):
            return params
        is_bool = isinstance(psi, (bool, np.bool_))     # the constructor refuses it
        if not is_bool and cls._psi_error(tau_f, psi_f) is None:
            return params
        if not (np.isfinite(tau_f) and 0.0 < tau_f < 1.0):
            return params
        return {**params, "psi": cls.repaired_psi(tau_f)}

    @classmethod
    def constrain_params(cls, params: dict) -> dict:
        """ψ clipped into ``EXTRA_PARAM_BOUNDS_BY_PARAM['psi']``, then made
        admissible with τ by :meth:`constructible_params`."""
        from pmcprg.copulas._base import EXTRA_PARAM_BOUNDS_BY_PARAM
        lo, hi, init = EXTRA_PARAM_BOUNDS_BY_PARAM["psi"]
        out = dict(params)
        out["psi"] = float(np.clip(float(out.get("psi", init)), lo, hi))
        return dict(cls.constructible_params(out))

    def _update_params(self):
        tau = float(self.params['tau_k'])
        psi = self.params.get('psi', _PSI_DEFAULT)
        if isinstance(psi, (bool, np.bool_)):
            raise CopulaParameterError(f'{self.class_name}: psi={psi!r} is not a number.')
        try:
            psi = float(psi)
        except (TypeError, ValueError):
            raise CopulaParameterError(f'{self.class_name}: psi={psi!r} is not a number.') from None
        err = self._psi_error(tau, psi)
        if err is not None:
            raise CopulaParameterError(err)
        pu, pv = self._weights(psi)
        theta = _theta_of(tau, pu, pv, self.class_name)
        self.theta = theta
        self.psi = psi
        self._pu, self._pv = pu, pv
        self.params['psi'] = psi
        if theta >= _THETA_HI:
            # θ clamped: store the τ it realises (RB-10, as Galambos).
            self.params['tau_k'] = _tau_of(theta, pu, pv)

    # -- evaluation ------------------------------------------------------

    def _logpdf_k(self, ka, kb):
        return _tawn_logpdf(ka, kb, self.theta, self._pu, self._pv)

    def pdf(self, uv):
        u = np.float64(minmaxEPS(uv[0]))
        v = np.float64(minmaxEPS(uv[1]))
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            return float(np.exp(self._logpdf_k(np.log(u), np.log(v))))

    def pdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised closed-form PDF, ``exp`` of :meth:`logpdf_array` (no floor)."""
        return np.exp(self.logpdf_array(uv))

    def logpdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Native log-PDF in log space (module docstring)."""
        uv = np.asarray(uv, dtype=float)
        u = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            return self._logpdf_k(np.log(u), np.log(v))

    def cdf(self, uv):
        u = np.float64(minmaxEPS(uv[0]))
        v = np.float64(minmaxEPS(uv[1]))
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            c = np.exp(_tawn_log_cdf(np.log(u), np.log(v), self.theta, self._pu, self._pv))
        return float(np.clip(c, 0.0, 1.0))

    def cdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised closed-form CDF."""
        uv = np.asarray(uv, dtype=float)
        u = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            c = np.exp(_tawn_log_cdf(np.log(u), np.log(v), self.theta, self._pu, self._pv))
        return np.clip(c, 0.0, 1.0)

    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = ∂C(u,v)/∂u, in log space (module docstring)."""
        u = np.float64(minmaxEPS(u))
        v = np.float64(minmaxEPS(v))
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            h = np.exp(_tawn_logh(np.log(v), np.log(u), self.theta, self._pu, self._pv))
        return float(np.clip(h, 0.0, 1.0))

    _BISECT_STEPS = 68

    def inv_h_array(self, w: np.ndarray, u: np.ndarray) -> np.ndarray:
        """v with h(v|u) = w, by vectorised bisection on logit v (module docstring).

        ``w`` at or below ``h(ε|u)`` gives ε, at or above ``h(1 − ε|u)``
        gives 1 − ε — :meth:`CopulaVirt.inv_h`'s saturation convention.
        """
        w = np.asarray(w, dtype=float)
        u = np.asarray(u, dtype=float)
        if w.shape != u.shape:
            raise ValueError(f"inv_h_array: w {w.shape} and u {u.shape} must match.")
        shape = w.shape
        w = w.ravel()
        ka = np.log(np.clip(u.ravel(), EPS, ONE_MINUS_EPS))
        th, pu, pv = self.theta, self._pu, self._pv
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            lw = np.log(w)
            x_lo = math.log(EPS) - math.log1p(-EPS)
            x_hi = -x_lo
            lo = np.full(w.shape, x_lo)
            hi = np.full(w.shape, x_hi)
            below = lw <= _tawn_logh(np.log(EPS), ka, th, pu, pv)
            above = lw >= _tawn_logh(math.log(ONE_MINUS_EPS), ka, th, pu, pv)
            for _ in range(self._BISECT_STEPS):
                mid = 0.5 * (lo + hi)
                kb = -np.logaddexp(0.0, -mid)          # ln expit(mid)
                up = _tawn_logh(kb, ka, th, pu, pv) < lw
                lo = np.where(up, mid, lo)
                hi = np.where(up, hi, mid)
            v = 1.0 / (1.0 + np.exp(-0.5 * (lo + hi)))
        v = np.clip(v, EPS, ONE_MINUS_EPS)
        v = np.where(below, EPS, np.where(above, ONE_MINUS_EPS, v))
        return v.reshape(shape)

    def inv_h(self, w: float, u: float) -> float:
        """Scalar :meth:`inv_h_array`."""
        return float(self.inv_h_array(np.array([float(w)]), np.array([float(u)]))[0])

    def tail_dependence(self) -> tuple[float, float]:
        """λ_L = 0, λ_U = ψ − ((1 + ψ^θ)^{1/θ} − 1) (module docstring)."""
        th, psi = self.theta, self.psi
        return 0.0, float(psi - math.expm1(math.log1p(psi ** th) / th))

    # -- fitting ---------------------------------------------------------

    @classmethod
    def fit(cls, data: np.ndarray, method: str = 'mle'):
        """Joint MLE of (τ, ψ); ``method='tau'`` falls back to it (module docstring)."""
        if method == 'tau':
            logger.warning(
                "%s.fit: method='tau' cannot identify psi from Kendall's tau "
                "alone; falling back to MLE.", cls.__name__)
            method = 'mle'
        return super().fit(data, method=method)


class CopulaTawn1(_CopulaTawn):
    """Tawn type 1: ψ_u = 1, ψ_v = ψ — ``ℓ = (1 − ψ) z + [w^θ + (ψ z)^θ]^{1/θ}``.

    See :class:`_CopulaTawn` for the parameters and the module docstring for
    the model, the τ < ψ constraint and the convention.
    """
    _FIXED_U = True


class CopulaTawn2(_CopulaTawn):
    """Tawn type 2: ψ_u = ψ, ψ_v = 1 — the transpose of type 1.

    See :class:`_CopulaTawn` for the parameters and the module docstring for
    the model, the τ < ψ constraint and the convention.
    """
    _FIXED_U = False


if __name__ == '__main__':
    from pathlib import Path

    for klass in (CopulaTawn1, CopulaTawn2):
        cop = klass(tau_k=0.4, psi=0.6)
        print(f'Copula   : {cop.copula_enum.value.LONG_NAME}')
        print(f'tau, psi : {cop.params["tau_k"]:.4f}, {cop.psi:.4f}   theta = {cop.theta:.6f}')
        print(f'C(0.3, 0.7) = {cop.cdf([0.3, 0.7]):.6f}   C(0.7, 0.3) = {cop.cdf([0.7, 0.3]):.6f}')
        print(f'tail dep : {cop.tail_dependence()}')

    plot_dir = Path('./data/Plots/Copulas')
    plot_dir.mkdir(parents=True, exist_ok=True)
    cop = CopulaTawn1(tau_k=0.4, psi=0.6)
    cop.plot_pdf(plot_dir)
    cop.plot_samples(plot_dir)
