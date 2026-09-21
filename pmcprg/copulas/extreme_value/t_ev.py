"""
t-EV copula — the extreme-value limit of the Student-t copula (Demarta &
McNeil 2005; Nikoloulopoulos, Joe & Li 2009), the last of the four
extreme-value families of FR-9.

The model
---------
Let ρ ∈ (−1, 1) and ν > 0 be the correlation and degrees of freedom of a
Student-t copula, w = −ln u, z = −ln v, and in this package's convention
t = w/(w + z) (the u-share, as in ``galambos.py``, ``husler_reiss.py`` and
``tawn.py``). With T_n, t_n the Student-t CDF and density with n = ν + 1
degrees of freedom and k = √((ν + 1)/(1 − ρ²)),

    ℓ(w, z) = w·T_n(a) + z·T_n(b),   C(u, v) = exp(−ℓ(w, z)),
    a = k·((w/z)^{1/ν} − ρ),         b = k·((z/w)^{1/ν} − ρ),
    A(t) = ℓ(t, 1 − t) = t·T_n(k((t/(1−t))^{1/ν} − ρ))
                         + (1 − t)·T_n(k(((1−t)/t)^{1/ν} − ρ)).

*Derived, not transcribed.* For an EV copula, ℓ_w(w, z) =
lim_{s→0} P(V ≤ 1 − s z | U = 1 − s w). For the Student-t copula that
conditional probability is the h-function
T_n((y − ρx)·√((ν + 1)/((ν + x²)(1 − ρ²)))), x, y the t_ν quantiles of
1 − s w, 1 − s z; as s → 0, x, y → ∞ with y/x → (w/z)^{1/ν} (the t_ν tail
is ∝ x^{−ν}) and ν + x² ~ x², so ℓ_w = T_n(k((w/z)^{1/ν} − ρ)) = T_n(a).
By symmetry ℓ_z = T_n(b), and Euler's relation for the degree-1 function ℓ
gives ℓ = wℓ_w + zℓ_z — the formula above. Checked in ``mpmath`` (the
Student h-limit at s = 10⁻⁴⁰ matches T_n(a) to 12 digits) and, with the
package's own ``CopulaStudent``, in the tests (the gap closes like s^{2/ν}).

*The brief's form is right, in this convention.* Its argument assignment
((t/(1−t))^{1/ν} in the t-term) is forced: with the other one, A(1) =
T_n(−kρ) ≠ 1 (0.127 at ρ = 0.5, ν = 4). The family is exchangeable by
construction — A(t) = A(1 − t), C(u, v) = C(v, u) — so no u/v convention
has to be chosen, unlike Tawn.

Checked at 400 random 40-digit ``mpmath`` points (t, ρ, ν ∈ [0.05, 10⁴]):
max(t, 1 − t) ≤ A ≤ 1, A'' ≥ 0 and A(t) = A(1 − t) — no violation.
A(0) = A(1) = 1 because (t/(1−t))^{1/ν} → ∞ and → 0 at the ends.

λ_U = 2(1 − A(½)) = 2·T_n(−k(1 − ρ)) = 2·T_n(−√((ν + 1)(1 − ρ)/(1 + ρ))),
λ_L = 0 — the Student-t copula's own upper-tail coefficient, as it must be
(the EV limit keeps the tail). Checked against 2(1 − A(½)) in ``mpmath``
(≤ 7·10⁻¹⁶ relative).

Hüsler–Reiss limit: with x = √((ν + 1)(1 − ρ)/(1 + ρ)) held at 1/λ, a =
k(1 − ρ) + k ln(w/z)/ν + O(1/ν²) → 1/λ + (λ/2) ln(w/z) (k/ν → λ/2) and
T_n → Φ, so ν → ∞ gives Hüsler–Reiss(λ). Measured at τ_HR ∈ {0.1, 0.5,
0.9}: max |ΔC| on the edge grid 6·10⁻⁴, 6·10⁻⁵, 6·10⁻⁶ at ν = 10², 10³,
10⁴ (|Δτ| 2.4·10⁻⁴ … 1.3·10⁻⁵ at ν = 10³), i.e. rate 1/ν; ln c converges at
the same rate where it is not a far tail (0.036 nat at ν = 10³), but only
pointwise — t tails are polynomial, normal ones are not.

Parametrisation and reachable τ
-------------------------------
The package's parameters are τ plus one extra. The extra is ν — as for
``CopulaStudent`` — and ρ is recovered from (τ, ν). Internally the
dependence parameter is s = ln η, η = (1 − ρ)/(1 + ρ) (ρ = −tanh(s/2)):
1 − ρ and 1 + ρ both keep full relative precision, which ρ itself does not
at the ends (τ = 1 − 10⁻⁶ at ν = 100 needs 1 − ρ ≈ 10⁻¹⁴).

*Every τ ∈ (0, 1) is reached at every ν — there is no joint constraint.*
ρ → 1: k → ∞ and a → ±∞ on either side of t = ½, A → max(t, 1 − t) (τ → 1).
ρ → −1: k → ∞ with a, b ≥ k|ρ| → +∞ everywhere, A → 1 (τ → 0). So negative
ρ is admissible — it is still an EV copula, with weak positive dependence
(τ = 0.0087 at ρ = −0.5, ν = 4) — and τ = 0 is never attained. τ is
decreasing in s on every grid tried (ν ∈ [0.05, 10⁴], s' = ln((ν + 1)η) ∈
[ln 10⁻¹⁴, 200]; no proof attempted). Hence ``constructible_params`` stays
the identity, and ν is *not identified from τ* (one τ, a whole curve of
(ρ, ν)): ``fit(method='tau')`` is itau — τ̂, then ν by maximum likelihood
at that τ (FR-12; it used to warn and fall back to MLE).

λ_U brackets τ: for any symmetric EV copula, τ = ∫t(1−t)A''/A with
½ ≤ A ≤ 1 and ∫t(1−t)A'' = 2∫(1 − A) (by parts); 1 − A is concave with
maximum 1 − A(½) = λ_U/2 at ½, so λ_U/4 ≤ ∫(1 − A) ≤ λ_U/2, i.e.
λ_U/2 ≤ τ ≤ 2λ_U. Numerically even τ ≤ λ_U (ratio τ/λ_U ∈ [0.53, 1)).
The bracket seeds the τ → s Brent search, and makes λ_U a fitting
coordinate (below).

*The numerical cap* is ν-free: ``_TAU_CAP`` = 1 − 1.25·10⁻⁶. A larger τ is
built at the s where τ(s, ν) = ``_TAU_CAP`` and stores it (RB-10). There x
lies in [1.00, 1.14]·10⁻⁶ for ν ∈ [0.05, 10³] (the Hüsler–Reiss λ = 1/x ≈
10⁶ of that module's cap; 1 − τ at x = 10⁻⁶ ranges from 1.10·10⁻⁶ at ν ≈ 2
to 1.245·10⁻⁶ at ν = 0.05, and 1 − τ is proportional to x there). The low
end never clamps: τ(s' = 1000) < 10⁻²¹⁸ even at ν = 10⁻³. ``theta`` holds
ρ for display and for the reference file of ``test_copula_limits.py``; it
rounds to ±1 at the far ends, where s does not.

*Admissible ν*: [0.05, 10³] — the range on which ln T_n, the kernel and τ
were checked against ``mpmath``. Above ν ≈ 2000 ``stdtr`` underflows
before the series below applies; ν → ∞ is the Hüsler–Reiss family, which
the package has. The fitting box is ν ∈ [0.5, 100]
(``EXTRA_PARAM_BOUNDS_BY_PARAM["nu"]``, init 4, Student's default).

*Why ``nu`` and not ``df``.* ``_fit._two_parameter_spec`` dispatches on the
extra parameter's *name*, ``EXTRA_PARAM_BOUNDS_BY_PARAM`` is keyed by name,
and so are the GUI's extra-parameter widgets. Under ``df`` t-EV would have
taken Student's branch — (atanh τ, 1/ν), which would still have fitted,
through a Brent inversion per likelihood evaluation — with Student's
bounds ν > 2.001, which t-EV does not need, and shared one GUI widget
with Student. ``nu`` gets its own bounds and its own branch.

Densities
---------
Differentiating ℓ_w = T_n(a) in z (a depends on r = w/z through
q = r^{1/ν}):

    ℓ_wz = −t_n(a)·k·q/(ν z) ≤ 0,   and equally  = −t_n(b)·k/(q ν w),

the two forms being equal because t_n(b)/t_n(a) = q^{ν+2} (both
n + a² = k²(1 + q² − 2ρq) and n + b² = k²q⁻²(1 + q² − 2ρq)). This is also
why the cross terms of ∂ℓ/∂w cancel, confirming ℓ_w = T_n(a). With
∂/∂u = −(1/u)∂/∂w (``husler_reiss.py``) and m = −ℓ_wz ≥ 0:

    h(v|u) = ∂C/∂u = C·ℓ_w/u,        c(u, v) = C·(ℓ_w ℓ_z + m)/(uv),
    ln h = w·T_n(−a) − z·T_n(b) + ln T_n(a),
    ln c = w·T_n(−a) + z·T_n(−b) + ln(T_n(a)T_n(b) + m),

using w + z − ℓ = w(1 − T_n(a)) + z(1 − T_n(b)) — every term is a sum of
non-negative quantities, nothing cancels. For τ, Euler's relation gives
A''(t) = m(t, 1 − t)/(t(1 − t)).

Numerics
--------
* a = ½√(ν+1)·[(q − 1)/√η + (q + 1)√η], q − 1 = expm1(ln(w/z)/ν): ρ is
  never formed, and the genuine sign change of a (q = ρ) is resolved to
  absolute precision.
* m uses the a-form where w ≤ z and the b-form where w > z: the argument of
  t_n then lies in (−kρ, k(1 − ρ)] and q^{±1} ≤ 1 — no overflow for
  ν = 0.05, where ln(w/z)/ν reaches ±780.
* ln T_n(x): ``log1p(−stdtr(n, −x))`` for x > 0; for x < −√n the series
  T_n(x) = ½ I_y(n/2, ½), I_y(p, ½) = y^p (1 − y)^½ ₂F₁(p + ½, 1; p + 1; y)
  /(p B(p, ½)), y = n/(n + x²) < ½ (DLMF 8.17.8), with ln y and ln(1 − y)
  from ln|x| and log1p(n/x²); otherwise ``log(stdtr)``, which does not
  underflow there as long as ln T_n(−√n) ≈ −0.35 n stays above −745
  (n ≲ 2000; with the cut at y < 0.1 instead, n = 1001 already had a gap).
  Against 50-digit ``mpmath``, n ∈ [1.05, 1001], x ∈ [−10⁶, 10⁶]: ≤ 7·10⁻¹⁶
  relative. ln t_n uses −ln B(n/2, ½) (no Γ-ratio cancellation) and
  Student's large-x form of log1p(x²/n).
* τ is not computed with ``_pickands.tau_from_A_terms_gl``: in t the
  integrand has a ridge of width ≈ ν/(4k) at ½ and, for ρ > 0, a second
  feature at t* = σ(ν ln ρ), as small as e^{−70} at ν = 100; the shared
  t-space rule centred at ½ is off by up to 1.8·10⁻⁹ relative on small τ
  and 10⁻¹² near τ = 1 (``quad``: 100 % at ν = 100, s' = −20). Here the
  Genest–MacKay integral is taken in L = logit t, where
  t(1 − t) A''/A dt = m·t(1 − t)/A dL, and on L ≤ 0 by symmetry:

      τ = 2 ∫_{−∞}^0 t_n(a)·k·e^{L/ν}·σ(L) / (ν·A) dL,   t = σ(L),

  evaluated in log space (the integrand decays like e^{(1+1/ν)L}) by a
  16-point composite Gauss–Legendre rule on [−1024, 0] with a doubling
  grid and ×3-graded grids around L = 0 and around the zero L* = ν ln ρ of
  a (ρ > 0). Against a 30-digit ``mpmath`` quadrature in L whose A', A''
  are numerical derivatives of the Demarta–McNeil A (not this module's
  closed forms), 36 points ν ∈ [0.5, 100], τ ∈ [10⁻²¹, 1 − 5·10⁻⁵]:
  ≤ 7·10⁻¹⁵ relative, at 0.2 ms per τ. A Monte-Carlo Kendall's τ
  (16 × 50 000 pairs at (τ, ν) = (0.2, 1), (0.4, 0.5), (0.5, 4), (0.8, 50))
  agrees within 1.3 standard errors (the common sign of the four gaps comes
  from their shared seeds: a Gaussian copula on the same seeds is 0.98 s.e.
  low too).
* τ → s: Brent on s, bracketed by the λ_U inequalities (closed-form
  s(λ_U) from ``stdtrit``), xtol 10⁻¹³; a memo of the forward map hands the
  constructor the very s the joint MLE produced (as ``tawn.py``).
* ``inv_h`` has no closed form: vectorised bisection on logit v (68
  halvings), as ``tawn.py``.

Validation of the kernel: against ``mpmath`` — C from ℓ, h and c by
numerical differentiation of C in (ln w, ln z) at 40–640 digits (raised
until two precisions agree to 10⁻¹⁴) — on 1744 points (u, v ∈ {10⁻¹², 10⁻⁶,
0.01, 0.3, 0.5, 0.9, 1 − 10⁻⁶, 1 − 10⁻¹²}, 29 (τ, ν) with ν ∈ {0.05, 0.5, 1,
2, 4, 50, 100, 10³}, τ from 10⁻¹⁰ to 0.999): largest relative errors
3.5·10⁻¹³ (pdf), 1.1·10⁻¹⁴ (cdf), 7.6·10⁻¹⁴ (h). Where the differentiation
needs more digits (ln c < −250, 112 points), the closed forms evaluated in
60-digit ``mpmath`` (validated by the former wherever both apply, to
8·10⁻⁵¹): pdf within 1.4·10⁻¹² relative (ν = 10³, ln c down to −6308),
cdf 1.1·10⁻¹⁴; h agrees except where it is below 10⁻³⁰⁸ and returns 0.

Fitting
-------
``fit(method='mle')`` is the joint MLE of
:func:`pmcprg.copulas._fit._fit_two_parameter_mle` in the coordinates
(logit λ_U, 1/ν) — a box every point of which is a t-EV copula, with s in
closed form from (λ_U, ν), so no Brent search per evaluation; λ_U ∈
[10⁻⁶, τ_hi] keeps τ ∈ [5·10⁻⁷, τ_hi]. As for Tawn the box is run twice
(a restart). Recovery at n = 3000 (20 replicates, all converged, 0.15 s per
fit): τ RMSE ≤ 0.012 everywhere; ν RMSE 0.10 at (τ, ν) = (0.3, 1), 0.08 at
(0.7, 1), 0.22 at (0.5, 2), 0.35 at (0.9, 3), 0.79 at (0.7, 5), 1.2 at
(0.2, 4); **large ν is weakly identified** — at (0.5, 10) ν̂ ranges over
[6.0, 74.6] (RMSE 14.9, median 9.8), and at (0.5, 50) over [16, 100], half
the fits at the box end 100 (median 73). Five contrived starts (τ_start
from 10⁻⁶ to 0.95) reach the same optimum within 10⁻¹² nat on five data
sets. Standard errors are not implemented (``NotImplementedError``, as for
Tawn).

References
----------
* Demarta, S. & McNeil, A. J. (2005). The t copula and related copulas.
  *International Statistical Review* 73(1), 111–129,
  doi:10.1111/j.1751-5823.2005.tb00254.x (the t-EV copula, §4).
* Nikoloulopoulos, A. K., Joe, H. & Li, H. (2009). Extreme value properties
  of multivariate t copulas. *Extremes* 12(2), 129–148,
  doi:10.1007/s10687-008-0072-4.
* Joe, H. (2014). *Dependence Modeling with Copulas*, Chapman & Hall/CRC,
  §6.1 (EV-copula identities) and §4.16 (t-EV).
* Genest, C. & MacKay, J. (1986). The joy of copulas: bivariate
  distributions with uniform marginals. *The American Statistician* 40(4),
  280–283.
* DLMF §8.17 (incomplete Beta function), https://dlmf.nist.gov/8.17.
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
from scipy.special import betaln, hyp2f1, stdtr, stdtrit

from pmcprg.copulas._base import CopulaVirt
from pmcprg.copulas.elliptical.student import _log1p_sq_over
from pmcprg.exceptions import CopulaParameterError
from pmcprg.numerics import EPS, ONE_MINUS_EPS, minmaxEPS

logger = logging.getLogger(__name__)

_LOG2 = math.log(2.0)

_NU_DEFAULT = 4.0          # a block without nu: Student's default ν
_NU_MIN = 0.05             # admissible ν range: the range the kernel and τ
_NU_MAX = 1.0e3            # were checked on against mpmath (module docstring)

# Upper cap of the reachable τ, the same for every ν (module docstring,
# "Reachable τ"): a τ above builds at the s where τ(s, ν) = _TAU_CAP.
_TAU_CAP = 1.0 - 1.25e-6
# Brent bracket for s = ln η, as s' = ln((ν + 1)·η) = 2 ln x: x = 10⁻⁷ lies
# beyond the cap for every admissible ν (1 − τ ≈ 1.1–1.3·10⁻⁷ there), and
# τ(s' = 1000) < 10⁻²¹⁸ < ε at ν = 10⁻³ already.
_SP_LO = 2.0 * math.log(1e-7)
_SP_HI = 1000.0


# ---------------------------------------------------------------------------
# Student-t CDF and density in log space (module docstring, "Numerics")
# ---------------------------------------------------------------------------

_SERIES_Y = 0.5     # y = n/(n + x²) below which ln T_n(x), x < 0, uses the series


def _log_tcdf(n, x):
    """ln T_n(x) for every x, including far lower tails where ``stdtr`` underflows.

    x > 0: ``log1p(−T_n(−x))``. x ≤ 0 with y = n/(n + x²) < ½: the
    incomplete-Beta series T_n(x) = ½·I_y(n/2, ½),
    I_y(a, b) = y^a (1 − y)^b / (a B(a, b)) · ₂F₁(a + b, 1; a + 1; y)
    (DLMF 8.17.8), whose ₂F₁ is 1 + O(y); ln y and ln(1 − y) are formed from
    ln|x| and log1p(n/x²). Otherwise ``log(stdtr)``.
    """
    x = np.asarray(x, dtype=float)
    shape = x.shape
    x = x.ravel()
    out = np.empty(x.shape)
    pos = x > 0.0
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        out[pos] = np.log1p(-stdtr(n, -x[pos]))
        neg = ~pos
        xn = x[neg]
        x2 = xn * xn
        ser = x2 * _SERIES_Y > n * (1.0 - _SERIES_Y)             # y < _SERIES_Y
        l1p = np.log1p(n / np.where(ser, x2, 1.0))                # −ln(1 − y)
        log_y = math.log(n) - 2.0 * np.log(np.abs(np.where(ser, xn, 1.0))) - l1p
        a = 0.5 * n
        series = (-_LOG2 + a * log_y - 0.5 * l1p - math.log(a) - betaln(a, 0.5)
                  + np.log(hyp2f1(a + 0.5, 1.0, a + 1.0, np.where(ser, np.exp(log_y), 0.0))))
        out[neg] = np.where(ser, series, np.log(stdtr(n, xn)))
    return out.reshape(shape)


def _log_tpdf(n, x):
    """ln t_n(x) = −ln B(n/2, ½) − ½ ln n − ((n + 1)/2)·ln(1 + x²/n) — the
    Beta form of the Γ ratio does not cancel at large n."""
    with np.errstate(over='ignore'):
        return -betaln(0.5 * n, 0.5) - 0.5 * math.log(n) - 0.5 * (n + 1.0) * _log1p_sq_over(x, n)


# ---------------------------------------------------------------------------
# Kernel (module docstring, "Densities" and "Numerics"). The dependence
# parameter is s = ln η, η = (1 − ρ)/(1 + ρ); L = ln(w/z).
# ---------------------------------------------------------------------------

def _log_k(s, nu):
    """ln k, k = √((ν + 1)/(1 − ρ²)) = ½√(ν + 1)·(η^{−1/2} + η^{1/2})."""
    return 0.5 * math.log(nu + 1.0) - _LOG2 + float(np.logaddexp(-0.5 * s, 0.5 * s))


def _args(L, s, nu):
    """``(a, b)``: a = k·(q − ρ), b = k·(1/q − ρ), q = e^{L/ν}, written as
    a = ½√(ν + 1)·[(q − 1)/√η + (q + 1)·√η] — ρ never formed, q − 1 by expm1."""
    c0 = 0.5 * math.sqrt(nu + 1.0)
    ie, se = math.exp(-0.5 * s), math.exp(0.5 * s)
    with np.errstate(over='ignore', invalid='ignore'):
        ea = np.expm1(L / nu)
        eb = np.expm1(-L / nu)
        a = c0 * (ea * ie + (2.0 + ea) * se)
        b = c0 * (eb * ie + (2.0 + eb) * se)
    return a, b


def _parts(w, z, s, nu):
    """``(a, b, log_lw, log_lz, log_m)`` at (w, z) > 0: the arguments, ln ℓ_w,
    ln ℓ_z and ln m, m = −ℓ_wz ≥ 0.

    m = t_{ν+1}(a)·k·q/(ν z) = t_{ν+1}(b)·k/(q ν w) (the two forms are equal,
    module docstring); the first is used where L ≤ 0, where a lies in
    (−kρ, k(1 − ρ)], the second where L > 0, where b does — the argument of
    the density stays bounded and q^{±1} ≤ 1.
    """
    lw, lz = np.log(w), np.log(z)
    L = lw - lz
    a, b = _args(L, s, nu)
    n = nu + 1.0
    use_a = L <= 0.0
    with np.errstate(over='ignore', invalid='ignore'):
        log_m = (_log_tpdf(n, np.where(use_a, a, b)) + _log_k(s, nu) - math.log(nu)
                 + np.where(use_a, L / nu - lz, -L / nu - lw))
    return a, b, _log_tcdf(n, a), _log_tcdf(n, b), log_m


def _tev_logpdf(ka, kb, s, nu):
    """ln c = w·T(−a) + z·T(−b) + ln(ℓ_w ℓ_z + m), from ``ka = ln u``, ``kb = ln v``."""
    w, z = -ka, -kb
    a, b, log_lw, log_lz, log_m = _parts(w, z, s, nu)
    n = nu + 1.0
    return w * stdtr(n, -a) + z * stdtr(n, -b) + np.logaddexp(log_lw + log_lz, log_m)


def _tev_log_cdf(ka, kb, s, nu):
    """ln C = −ℓ = −(w·T(a) + z·T(b))."""
    w, z = -ka, -kb
    a, b = _args(np.log(w) - np.log(z), s, nu)
    n = nu + 1.0
    return -(w * stdtr(n, a) + z * stdtr(n, b))


def _tev_logh(kb, ka, s, nu):
    """ln h(v | u) = w·T(−a) − z·T(b) + ln T(a), from ``kb = ln v``, ``ka = ln u``."""
    w, z = -ka, -kb
    a, b = _args(np.log(w) - np.log(z), s, nu)
    n = nu + 1.0
    return w * stdtr(n, -a) - z * stdtr(n, b) + _log_tcdf(n, a)


def _tev_A_terms(t, s, nu):
    """``(A, A', A'')`` at ``t`` (array), closed form: A = t T(a) + (1 − t) T(b),
    A' = T(a) − T(b), A'' = m(t, 1 − t)/(t(1 − t)) (module docstring)."""
    t = np.asarray(t, dtype=float)
    tm = 1.0 - t
    n = nu + 1.0
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        a, b, _, _, log_m = _parts(t, tm, s, nu)
        Ta, Tb = stdtr(n, a), stdtr(n, b)
        App = np.exp(log_m - np.log(t) - np.log(tm))
    App = np.where(np.isfinite(App), App, 0.0)
    return t * Ta + tm * Tb, Ta - Tb, App


def _lambda_u(s, nu):
    """λ_U = 2·T_{ν+1}(−x), x = √((ν + 1)·η) = √((ν + 1)(1 − ρ)/(1 + ρ))."""
    return float(2.0 * stdtr(nu + 1.0, -math.exp(0.5 * (s + math.log(nu + 1.0)))))


def _s_from_lambda(lam, nu):
    """The s with λ_U(s, ν) = ``lam`` ∈ (0, 1): x = −T_{ν+1}⁻¹(λ/2)."""
    x = -float(stdtrit(nu + 1.0, 0.5 * lam))
    return 2.0 * math.log(x) - math.log(nu + 1.0)


# ---------------------------------------------------------------------------
# τ(s, ν) — composite Gauss–Legendre in L = logit t (module docstring)
# ---------------------------------------------------------------------------

_GL_X, _GL_W = np.polynomial.legendre.leggauss(16)
_L_MIN = -1024.0


def _l_mesh(s, nu):
    """Breakpoints on [_L_MIN, 0]: a doubling grid 0.125·2^j, and grids
    ``c ± width·3^j`` around the two features of the integrand — L = 0,
    where a = k(1 − ρ), and the zero L* = ν ln ρ of a when ρ > 0."""
    pts = {0.0, _L_MIN}
    step = 0.125
    while step < -_L_MIN:
        pts.add(-step)
        step *= 2.0
    k = math.exp(_log_k(s, nu))
    rho = -math.tanh(0.5 * s)
    a0 = math.sqrt(nu + 1.0) * math.exp(0.5 * s)            # a at L = 0: k(1 − ρ)
    features = [(0.0, (nu / k) * max(1.0, a0 / (nu + 2.0)))]
    if rho > 0.0 and nu * math.log(rho) > _L_MIN:
        features.append((nu * math.log(rho), nu / (k * rho)))
    for c, width in features:
        st = width
        while st < -_L_MIN:
            pts.add(c - st)
            pts.add(c + st)
            st *= 3.0
    return np.array(sorted(p for p in pts if _L_MIN <= p <= 0.0))


def _tev_tau_quad(s, nu):
    """τ(s, ν) = 2 ∫_{−∞}^0 m·t(1 − t)/A dL, t = σ(L) (module docstring)."""
    b = _l_mesh(s, nu)
    lo, hi = b[:-1, None], b[1:, None]
    L = (0.5 * (hi - lo) * _GL_X + 0.5 * (hi + lo)).ravel()
    wt = (0.5 * (hi - lo) * _GL_W).ravel()
    n = nu + 1.0
    a, bb = _args(L, s, nu)
    log_t = -np.logaddexp(0.0, -L)          # ln σ(L)
    log_1mt = -np.logaddexp(0.0, L)         # ln σ(−L)
    with np.errstate(under='ignore'):
        A = np.exp(log_t) * stdtr(n, a) + np.exp(log_1mt) * stdtr(n, bb)
        f = np.exp(_log_tpdf(n, a) + _log_k(s, nu) + L / nu + log_t
                   - math.log(nu) - np.log(A))
    return float(np.clip(2.0 * np.dot(wt, f), 0.0, 1.0))


# Memo of the forward map and of its exact inverse on the values it produced
# (as ``tawn.py``): the joint MLE hands the constructor τ(s, ν), which would
# otherwise be re-inverted by Brent's method at every likelihood evaluation.
_MEMO_SIZE = 1 << 14
_TAU_MEMO: "OrderedDict[tuple, float]" = OrderedDict()
_S_MEMO: "OrderedDict[tuple, float]" = OrderedDict()


def _remember(memo, key, value):
    memo[key] = value
    memo.move_to_end(key)
    if len(memo) > _MEMO_SIZE:
        memo.popitem(last=False)


def _tau_of(s, nu):
    """τ(s, ν), memoised; records s as the inverse of the returned τ on every call."""
    key = (float(s), float(nu))
    tau = _TAU_MEMO.get(key)
    if tau is None:
        tau = _tev_tau_quad(*key)
        _remember(_TAU_MEMO, key, tau)
    if tau > 0.0:
        _remember(_S_MEMO, (tau, key[1]), key[0])
    return tau


def _s_of(tau, nu, family_name="CopulaTEV"):
    """s with τ(s, ν) = τ (Brent on s); τ above ``_TAU_CAP`` is solved at the cap."""
    tau = float(tau)
    if tau <= 0.0:
        raise CopulaParameterError(
            f'{family_name}: tau_k={tau!r} must be > 0 (no negative or zero '
            f'dependence in an extreme-value copula).')
    tau = min(tau, _TAU_CAP)
    nu = float(nu)
    hit = _S_MEMO.get((tau, nu))
    if hit is not None:
        return hit
    ln1 = math.log(nu + 1.0)

    def f(s):
        return _tau_of(s, nu) - tau

    # λ_U/2 ≤ τ ≤ 2λ_U for every symmetric EV copula (module docstring):
    # the root lies between the s of λ_U = min(4τ, ·) and of λ_U = τ/4,
    # widened, else the whole bracket.
    lo, hi = _SP_LO - ln1, _SP_HI - ln1
    try:
        lo_b = _s_from_lambda(4.0 * tau, nu) - 0.5 if 4.0 * tau < 1.0 else lo
        hi_b = _s_from_lambda(0.25 * tau, nu) + 0.5
        if np.isfinite(lo_b) and np.isfinite(hi_b) and lo <= lo_b < hi_b <= hi \
                and f(lo_b) > 0.0 > f(hi_b):
            lo, hi = lo_b, hi_b
    except (ValueError, OverflowError):
        pass
    if f(hi) >= 0.0:            # cannot happen for ν ≥ 10⁻³ (bracket comment)
        return hi
    s = brentq(f, lo, hi, xtol=1e-13, rtol=4.0 * np.finfo(float).eps, maxiter=200)
    _remember(_S_MEMO, (tau, nu), s)
    return s


# ---------------------------------------------------------------------------
# Copula class
# ---------------------------------------------------------------------------

class CopulaTEV(CopulaVirt):
    """t-EV copula — the extreme-value limit of the Student-t copula.

    Parameters
    ----------
    tau_k : float
        Kendall's τ ∈ (0, 1). Every τ is reachable at every ν; above
        ``_TAU_CAP`` = 1 − 1.25·10⁻⁶ the copula is built at the cap and
        stores it (RB-10).
    nu : float, optional
        Degrees of freedom ν of the parent Student-t copula,
        0.05 ≤ ν ≤ 10⁴ (default 4.0, Student's default). Named ``nu``, not
        ``df``: see the module docstring ("Parametrisation").

    ``theta`` is the correlation ρ of the parent t copula (as for
    :class:`CopulaStudent`), recovered from (τ, ν).
    """

    n_params: int = 2

    def __init__(self, **kwargs):
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    @classmethod
    def reachable_tau_bounds(cls) -> tuple[float, float]:
        """``(EPS, _TAU_CAP)`` — the same for every ν (module docstring)."""
        return float(EPS), _TAU_CAP

    @classmethod
    def _nu_error(cls, nu):
        if isinstance(nu, (bool, np.bool_)):
            return f'{cls.__name__}: nu={nu!r} is not a number.'
        try:
            nu = float(nu)
        except (TypeError, ValueError):
            return f'{cls.__name__}: nu={nu!r} is not a number.'
        if not (np.isfinite(nu) and _NU_MIN <= nu <= _NU_MAX):
            return f'{cls.__name__}: nu must be in [{_NU_MIN}, {_NU_MAX:g}], got {nu!r}.'
        return None

    def _update_params(self):
        tau = float(self.params['tau_k'])
        nu = self.params.get('nu', _NU_DEFAULT)
        err = self._nu_error(nu)
        if err is not None:
            raise CopulaParameterError(err)
        nu = float(nu)
        s = _s_of(tau, nu, self.class_name)
        self._s = s
        self.nu = nu
        self.theta = -math.tanh(0.5 * s)          # ρ of the parent t copula
        self.params['nu'] = nu
        if tau > _TAU_CAP:
            self.params['tau_k'] = _TAU_CAP      # RB-10: the τ it realises

    # -- evaluation ------------------------------------------------------

    def _logpdf_k(self, ka, kb):
        return _tev_logpdf(ka, kb, self._s, self.nu)

    def pdf(self, uv):
        u = np.float64(minmaxEPS(uv[0]))
        v = np.float64(minmaxEPS(uv[1]))
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            return float(np.exp(self._logpdf_k(np.log(np.array([u])), np.log(np.array([v])))[0]))

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
            c = np.exp(_tev_log_cdf(np.log(u), np.log(v), self._s, self.nu))
        return float(np.clip(c, 0.0, 1.0))

    def cdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised closed-form CDF."""
        uv = np.asarray(uv, dtype=float)
        u = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            c = np.exp(_tev_log_cdf(np.log(u), np.log(v), self._s, self.nu))
        return np.clip(c, 0.0, 1.0)

    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = ∂C(u,v)/∂u, in log space (module docstring)."""
        u = np.float64(minmaxEPS(u))
        v = np.float64(minmaxEPS(v))
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            h = np.exp(_tev_logh(np.log(np.array([v])), np.log(np.array([u])), self._s, self.nu)[0])
        return float(np.clip(h, 0.0, 1.0))

    _BISECT_STEPS = 68

    def inv_h_array(self, w: np.ndarray, u: np.ndarray) -> np.ndarray:
        """v with h(v|u) = w, by vectorised bisection on logit v (as ``tawn.py``).

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
        s, nu = self._s, self.nu
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            lw = np.log(w)
            x_lo = math.log(EPS) - math.log1p(-EPS)
            lo = np.full(w.shape, x_lo)
            hi = np.full(w.shape, -x_lo)
            below = lw <= _tev_logh(np.full(w.shape, math.log(EPS)), ka, s, nu)
            above = lw >= _tev_logh(np.full(w.shape, math.log(ONE_MINUS_EPS)), ka, s, nu)
            for _ in range(self._BISECT_STEPS):
                mid = 0.5 * (lo + hi)
                kb = -np.logaddexp(0.0, -mid)          # ln expit(mid)
                up = _tev_logh(kb, ka, s, nu) < lw
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
        """λ_L = 0, λ_U = 2·T_{ν+1}(−√((ν + 1)(1 − ρ)/(1 + ρ))) (module docstring)."""
        return 0.0, _lambda_u(self._s, self.nu)

    # -- fitting ---------------------------------------------------------

    @classmethod
    def fit(cls, data: np.ndarray, method: str = 'mle', *, weights=None,
            pseudo_obs: bool = False):
        """Joint MLE of (τ, ν) by default; ``method='tau'`` is itau.

        Kendall's τ alone cannot identify ν: ``'tau'`` inverts τ̂ and fits
        ν by maximum likelihood at that τ (a profile likelihood, FR-12 —
        before, it logged a warning and ran the joint MLE). ``weights`` and
        ``pseudo_obs`` as in :meth:`CopulaVirt.fit`.
        """
        return super().fit(data, method=method, weights=weights, pseudo_obs=pseudo_obs)


if __name__ == '__main__':
    from pathlib import Path

    cop = CopulaTEV(tau_k=0.4, nu=3.0)
    print(f'Copula   : {cop.copula_enum.value.LONG_NAME}')
    print(f'tau, nu  : {cop.params["tau_k"]:.4f}, {cop.nu:.4f}   rho = {cop.theta:.6f}')
    print(f'C(0.3, 0.7) = {cop.cdf([0.3, 0.7]):.6f}   h(0.7|0.3) = {cop.conditional_cdf(0.7, 0.3):.6f}')
    print(f'tail dep : {cop.tail_dependence()}')

    plot_dir = Path('./data/Plots/Copulas')
    plot_dir.mkdir(parents=True, exist_ok=True)
    cop.plot_pdf(plot_dir)
    cop.plot_samples(plot_dir)
