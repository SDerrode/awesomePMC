"""
BB6 copula (Joe-Gumbel) — upper-tail dependence only, two parameters (FR-9).

Generator and CDF (re-derived, not transcribed)
-----------------------------------------------
BB6 is the **outer power** (Gumbel) transform of Joe's generator:

    φ(t) = ( − ln( 1 − (1 − t)^θ ) )^δ ,      θ ≥ 1,  δ ≥ 1,

i.e. ``φ = ψ_Joe^δ`` with ``ψ_Joe(t) = −ln(1 − (1 − t)^θ)``
(``pmcprg.copulas.archimedean.joe``). Inverting it — ``s = φ(t)`` ⇒
``s^{1/δ} = −ln(1 − (1 − t)^θ)`` ⇒ ``(1 − t)^θ = 1 − e^{−s^{1/δ}}`` — gives

    φ^{-1}(s) = 1 − ( 1 − e^{−s^{1/δ}} )^{1/θ},

and with ū = 1 − u, v̄ = 1 − v, p = −ln(1 − ū^θ), q = −ln(1 − v̄^θ),
S = (p^δ + q^δ)^{1/δ}, G = 1 − e^{−S}, the Archimedean construction
``C = φ^{-1}(φ(u) + φ(v))`` reduces to

    C(u, v) = 1 − G^{1/θ} = 1 − [ 1 − exp( −(p^δ + q^δ)^{1/δ} ) ]^{1/θ}.

Checked against ``φ^{-1}(φ(u) + φ(v))`` evaluated independently at 60 digits
in ``mpmath``: identical to the last digit at every (θ, δ, u, v) tried. The
**complement** ū appears inside the generator — as in Joe's, not BB1's — so
the natural kernel coordinate here is ``ka = ln(1 − u)``, Joe's, not BB1's
``ln u`` (see "Kernel interface" below).

Sub-models (verified, both exact)
---------------------------------
* **δ = 1** → ``φ = ψ_Joe`` → the **Joe** copula at the same θ. With δ = 1,
  ``S = p + q`` and ``G = 1 − e^{−(p+q)} = 1 − (1 − ū^θ)(1 − v̄^θ) =
  ū^θ + v̄^θ − ū^θ v̄^θ``, which is Joe's own ``W``; every expression below
  collapses to ``joe.py``'s by hand, and numerically to 60 digits.
* **θ = 1** → ``φ(t) = (−ln t)^δ`` → the **Gumbel–Hougaard** copula at
  parameter δ (``p = −ln u``, ``G = 1 − e^{−S}``, ``C = 1 − G = e^{−S}``).
Both are *boundaries* of the admissible set (θ ≥ 1, δ ≥ 1), which is what
makes their likelihood-ratio tests one-sided (``_stderr._SUBMODEL_NESTING``).
θ = δ = 1 is independence.

Kendall's τ — a closed form, contrary to the usual claim
--------------------------------------------------------
τ of an Archimedean copula is ``1 + 4∫₀¹ φ/φ′`` (Genest & MacKay 1986). An
outer power leaves that integral scaled: ``(φ^δ)/(φ^δ)′ = φ/(δφ′)``, so

    τ_BB6(θ, δ) = 1 − ( 1 − τ_Joe(θ) ) / δ                                (★)

**exactly**, with no quadrature at all — the same identity that gives BB1
(the outer power of Clayton) its ``1 − 2/(δ(θ+2))``. Checked against a
60-digit ``mpmath`` Genest–MacKay quadrature of BB6's *own* generator at
θ ∈ {1, 1+10⁻⁷, 1.5, 2, 3.7, 8} × δ ∈ {1, 1.4, 2.6, 5}: relative agreement
0 to 9·10⁻¹⁹ (the quadrature's own limit, not the identity's), and
τ(1, δ) = 1 − 1/δ reproduces Gumbel's closed form to all 60 digits. No
Archimedean τ quadrature is therefore added to the package: (★) reuses
``joe._joe_tau_from_theta`` / ``joe._joe_theta_from_tau``, whose three
expansions are exact from θ = 1 (τ = 0) to θ → ∞ (audit RB-2), so BB6
inherits Joe's accuracy near independence instead of a new integral's.

Parametrisation and the joint (τ, δ) constraint
-----------------------------------------------
The package stores ``tau_k`` plus at most one extra parameter. **δ is the
extra one** and θ is recovered from (τ, δ) — BB1's choice, and the one (★)
makes exact: ``τ_Joe = 1 − δ(1 − τ)`` then ``θ = joe.θ(τ_Joe)``. (The
converse choice, θ extra and δ = (1 − τ_Joe(θ))/(1 − τ), is equally
well-defined but leaves the outer power — the parameter that moves τ at
fixed tail shape — implicit.) At fixed δ, τ is strictly increasing in θ
because τ_Joe is; at fixed θ it is strictly increasing in δ.

τ_Joe ∈ [0, 1) forces ``0 < δ(1 − τ) ≤ 1``: the admissible δ-interval at τ is

    δ ∈ [1, 1/(1 − τ)]      ⟺      τ ≥ 1 − 1/δ,

**closed at both ends** — unlike BB1's ``[1, 1/(1 − τ))``, whose upper end
θ = 0 is the degenerate Clayton boundary. Here δ = 1/(1 − τ) is θ = 1, a
perfectly good Gumbel copula, so :meth:`CopulaBB6.delta_max` returns the
largest *double* δ with ``δ·(1 − τ) ≤ 1`` (found by ``nextafter`` from
``1/(1 − τ)``, so the constructor's own floating-point test is the one that
decides) and :meth:`constructible_params` repairs a refused draw to it
without padding. Every τ ∈ (0, 1) is reachable — at δ = 1 (Joe) — so the
registered range is Joe's own ``[ε, 1]``; but no τ ≤ 1 − 1/δ is reachable at
a given δ, so (τ, δ) is *jointly* constrained exactly as BB1's (τ, δ) and
Tawn's (τ, ψ) are, and the same hooks are implemented. A block without
``delta6`` is the **Joe member δ = 1**, admissible at every registered τ
(Tawn's ``ψ = 1`` precedent), not BB1's 1.5, which would be inadmissible
below τ = 1/3.

One resolution limit is worth stating: a δ *one ulp below* ``delta_max``
makes ``τ_Joe = 1 − δ(1 − τ)`` come out as exactly ``ε/2``, and no θ > 1 is
representable as a double below ``τ_Joe(1 + ε) = 1.288·10⁻¹⁶``. Such a
τ_Joe is taken to be 0 — the Gumbel member — which moves τ by less than one
of its own ulps; see :data:`_TAU_JOE_FLOOR`, whose comment also records the
``ValueError`` Joe's Brent bracket raises there if it is not.

*Why the name ``delta6`` and not ``delta``.* ``_fit._two_parameter_spec``
dispatches on the **parameter name**: ``"delta"`` is BB1's branch, with
BB1's ``τ = 1 − 2/(δ(θ + 2))`` map, and BB6 routed through it would be
fitted with the wrong parametrisation — silently, since every (θ, δ) pair
is numerically plausible. This is the same reason t-EV's ν is registered as
``"nu"`` and not as Student's ``"df"`` (``t_ev.py``). ``test_bb6.py`` proves
BB6 does not take BB1's path.

Tail dependence
---------------
From ``1 − φ^{-1}(s) = (1 − e^{−s^{1/δ}})^{1/θ} ≈ s^{1/(θδ)}`` as s → 0,
λ_U = 2 − lim (1 − φ^{-1}(2s))/(1 − φ^{-1}(s)) gives

    λ_U = 2 − 2^{1/(θδ)},                λ_L = 0,

the lower one because ``φ^{-1}(s) ≈ e^{−s^{1/δ}}/θ`` as s → ∞, so
``φ^{-1}(2s)/φ^{-1}(s) = exp(−s^{1/δ}(2^{1/δ} − 1)) → 0`` for every δ ≥ 1
(δ = 1 included — Joe's λ_L = 0). Checked numerically: the diagonal ratio
``(1 − 2u + C(u,u))/(1 − u)`` at u = 1 − 10^{−k}, k = 6…30, settles on
2 − 2^{1/(θδ)} to 12 digits, and ``C(u,u)/u`` falls to 10⁻³⁰ at k = 30.
λ_U depends on (θ, δ) only through the product θδ, which λ_U alone
therefore cannot separate.

Densities (derivation)
----------------------
Only p depends on u and only q on v. With m = 1 − ū^θ (so ``ln m = −p``),
``∂p/∂u = −θ ū^{θ−1}/m``, ``∂S/∂u = S^{1−δ} p^{δ−1} ∂p/∂u`` and
``C = 1 − G^{1/θ}``:

    h(v|u) = ∂C/∂u = G^{1/θ−1} e^{−S} S^{1−δ} p^{δ−1} ū^{θ−1} / m,

    c(u, v) = θ (pq)^{δ−1} (ū v̄)^{θ−1} e^{p+q−S} G^{1/θ−2} S^{1−2δ} · B,
    B = S(1 − 1/θ) + S·G/θ + (δ − 1)·G.

B's three terms are each ≥ 0 **because** θ ≥ 1 and δ ≥ 1 — the density is
positive on the whole square only on the admissible set. At δ = 1 the third
vanishes and the expressions collapse to Joe's ``(θ − 1 + W)`` form; at
θ = 1 the first vanishes and they collapse to Gumbel's ``(S + δ − 1)``; at
θ = δ = 1, ln c = 0 identically (independence). All three checked by hand
and numerically.

Numerics
--------
Everything is evaluated from ``ka = ln(1 − u)``, ``kb = ln(1 − v)`` and
nothing is ever formed in linear scale. With ``z = −θ·ka ≥ 0``:

    ln p = ln( −ln1mexp(z) ),   ln1mexp(z) = ln(1 − e^{−z}),

Mächler's (2012) switch at ln 2 (reused from ``joe.py`` — a third copy of
that helper would be a place for the two to drift). The two ends are the
two cancellations of this family and each is handled by its own branch:

* **u → 0** (z → 0): ``1 − ū^θ`` is the whole of p and cancels to nothing if
  formed as ``1 − exp(θ ka)``; ``−expm1(θ ka)`` is exact.
* **u → 1** (z → ∞): ``ū^θ`` underflows for z ≳ 745, and ``ln1mexp`` then
  returns exactly 0, whose log is −∞. But p = e^{−z}(1 + e^{−z}/2 + …) there,
  so ``ln p = −z`` to a relative 10⁻³⁰⁴: the branch ``z > 700`` returns −z.
  Without it, ``CopulaBB6(tau_k=0.9, delta6=3).logpdf_array`` is NaN on the
  whole ``u = 1 − 10⁻¹²`` edge whenever θ ≳ 2000 — the Galambos round's
  failure mode, found here by the ``mpmath`` ground truth of ``test_bb6.py``.

``p + q − S`` (in ln c) and ``p − S`` (in ln h) both cancel to zero at δ = 1,
where S = p + q resp. S ≥ p exactly. Neither is formed as a difference:
with ``d = ln S − ln p ≥ 0``,

    p − S = −exp( ln S + ln1mexp(d) ),

exact at d = 0 (ln1mexp(0) = −∞ ⇒ 0) and equal to −S as d → ∞; and
``p + q − S`` is bracketed as ``(p − S) + q`` when p ≥ q and as
``p + (q − S)`` otherwise, so the bracket is always the *smaller* of p, q
(Tawn's rule). ``ln B`` is a two-fold ``logaddexp`` of its three ≥ 0 terms,
so ``ln(θ − 1) = −∞`` at θ = 1 and ``ln(δ − 1) = −∞`` at δ = 1 are absorbed
exactly rather than cancelled. Measured against the adaptive-precision
``mpmath`` ground truth of ``test_bb6.py`` (120 to 3840 digits, C from the
generator alone and c, h as its finite differences) on the 6×6 grid
u, v ∈ [10⁻¹², 1 − 10⁻¹²] at fourteen (τ, δ) from (10⁻⁶, 1) to (0.99, 1),
both sub-model limits included: **≤ 1.4·10⁻¹⁴ on ln c** (absolute, and
relative where |ln c| > 1), **≤ 2.9·10⁻¹⁴ relative on C**, **≤ 4.6·10⁻¹³
relative on h**. There is no fallback — a value that cannot be computed is
NaN, not ``EPS`` (audit RB-9).

*Kernel interface* (audit FR-8's shape). ``_kcoord(x) = ln(1 − x)`` and
``_kcoord_reflected(x) = ln x`` are Joe's, because BB6's generator, like
Joe's and unlike BB1's, is built from the complement 1 − t. BB6 registers no
rotation or survival wrapper of its own (FR-8 is closed), but the interface
costs nothing and is the shape such a wrapper would consume.

*Inverse h-function.* No closed form (as for BB1 and Tawn): ``ln h(v|u) =
ln w`` is solved by 68 steps of vectorised bisection on ``logit v`` over
``[logit ε, logit(1 − ε)]`` (final bracket < 10⁻¹⁹ in logit), which never
forms 1 − v in linear scale — ``ln(1 − v) = −logaddexp(0, logit v)`` — with
:meth:`CopulaVirt.inv_h`'s saturation convention at the two ends. Bisection
rather than Brent because sampling 20 000 pairs through the base class's
per-point Brent search takes minutes (Tawn's measurement).

Fitting
-------
``fit`` is always the joint two-parameter MLE
(:func:`pmcprg.copulas._fit._fit_two_parameter_mle`); τ alone cannot
identify δ, so ``method='tau'`` logs a warning and falls back to MLE (BB1's
and Tawn's precedent). ``_two_parameter_spec``'s ``delta6`` branch optimises
in ``(ln(θ − 1), ln δ)``, whose box **is** the admissible set — every θ ≥ 1,
δ ≥ 1 is a BB6 copula — with τ mapped back by (★); a (τ, δ) box would have
to project onto ``δ ≤ 1/(1 − τ)``, the flat-plateau failure of audit RB-8.

References
----------
* Joe, H. (2014). *Dependence Modeling with Copulas*, Chapman & Hall/CRC,
  ch. 4 (the two-parameter BB families; BB6 = outer power of Joe's
  ``B5``/Joe family, θ ≥ 1, δ ≥ 1).
* Joe, H. (1997). *Multivariate Models and Dependence Concepts*, Chapman &
  Hall, ch. 5 (family BB6 and its Joe / Gumbel–Hougaard sub-models).
* Genest, C. & MacKay, J. (1986). The joy of copulas: bivariate
  distributions with uniform marginals. *The American Statistician* 40(4),
  280–283 (τ = 1 + 4∫φ/φ′, used to derive (★)).
* Hofert, M., Mächler, M. & McNeil, A. J. (2012). Likelihood inference for
  Archimedean copulas in high dimensions under known margins.
  *J. Multivariate Anal.* 110, 133–150, doi:10.1016/j.jmva.2012.02.019
  (log-space evaluation of Archimedean densities).
* Mächler, M. (2012). *Accurately computing log(1 − exp(−|a|))*, vignette of
  the R package Rmpfr (the ln 2 switch, reused from ``joe.py``).
* Schepsmeier, U. et al. (2018). *VineCopula*, R package (family 8, BB6,
  same (θ, δ) convention).
"""
if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import logging
import math

import numpy as np

from pmcprg.copulas._base import CopulaVirt
from pmcprg.copulas.archimedean.joe import (
    _joe_tau_from_theta,
    _joe_theta_from_tau,
    _log1mexp,
)
from pmcprg.exceptions import CopulaParameterError
from pmcprg.numerics import EPS, ONE_MINUS_EPS, minmaxEPS

logger = logging.getLogger(__name__)

# Smallest τ_Joe that any *double* θ realises: the next double after 1 is
# 1 + ε, and τ_Joe(1 + ε) = 1.288·10⁻¹⁶. Below it no θ > 1 exists in double
# precision, and ``joe._joe_theta_from_tau`` — whose Brent bracket is
# ``[τ/(1 − τ), (1 + τ)/(1 − τ)]`` — raises ``ValueError: f(a) and f(b) must
# have different signs``, because ``1 + τ/(1 − τ)`` *rounds up* to 1 + ε,
# whose τ already exceeds the target. BB6 reaches that band routinely, not
# exceptionally: ``τ_Joe = 1 − δ(1 − τ)`` lands on ε/2 for every δ one ulp
# below :meth:`CopulaBB6.delta_max`, which is exactly what the multistart
# repair produces. The nearest θ a double can represent is then 1 itself — the
# Gumbel member — so that is what is used, moving τ by at most
# 1.288·10⁻¹⁶/δ, below one ulp of τ. Joe's own inversion is left untouched.
_TAU_JOE_FLOOR: float = _joe_tau_from_theta(float(np.nextafter(1.0, 2.0)))

# ln(1 − e^{−z}) returns exactly 0 once e^{−z} underflows (z ≳ 745), and its
# log is then −∞ instead of −z. Above this threshold ln p = −z to a relative
# e^{−700} ≈ 10⁻³⁰⁴ (module docstring, "Numerics").
_Z_ASYMPTOTIC: float = 700.0

# A block without ``delta6`` is the Joe member, admissible at every
# registered τ (module docstring, "Parametrisation").
_DELTA_DEFAULT: float = 1.0

# Bisection steps of the inverse h-function, on logit v over
# [logit ε, logit(1 − ε)] (width ≈ 72): 72/2⁶⁸ < 10⁻¹⁹.
_BISECT_STEPS: int = 68


# ---------------------------------------------------------------------------
# Log-space kernel on ka = ln(1 − u), kb = ln(1 − v) (module docstring)
# ---------------------------------------------------------------------------

def _log_or_ninf(x: float) -> float:
    """``ln x`` for x > 0, ``−∞`` at x ≤ 0 — ``math.log`` raises there.

    Used for ``ln(θ − 1)`` and ``ln(δ − 1)``, which ``logaddexp`` then
    absorbs exactly at the Gumbel (θ = 1) and Joe (δ = 1) boundaries.
    """
    return math.log(x) if x > 0.0 else -math.inf


def _bb6_logp(k, th):
    """``ln p``, ``p = −ln(1 − (1 − u)^θ) > 0``, from ``k = ln(1 − u) ≤ 0``.

    ``z = −θ k ≥ 0``; ``ln p = ln(−ln1mexp(z))`` except beyond
    :data:`_Z_ASYMPTOTIC`, where ``(1 − u)^θ`` underflows and ``p = e^{−z}``
    to a relative 10⁻³⁰⁴ (module docstring, "Numerics").
    """
    z = -th * np.asarray(k, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        lp = np.log(-_log1mexp(np.minimum(z, _Z_ASYMPTOTIC)))
        return np.where(z > _Z_ASYMPTOTIC, -z, lp)


def _bb6_log_g(lS):
    """``ln G = ln(1 − e^{−S})`` from ``ln S`` — accurate at both ends.

    ``ln1mexp(S)`` needs S in linear scale, which underflows to 0.0 for
    ``ln S < −745``: its log is then −∞ where the true ``ln G`` is ``ln S``.
    That is not an exotic corner — ``ln S ≈ −1068`` at (u, v) = (1 − 10⁻¹²,
    1 − 10⁻¹²) for τ = 0.95, δ = 1 (θ = 38.7), where the density is a
    perfectly ordinary e²⁹·⁹. Below ``ln S = −20`` (S < 2·10⁻⁹) the series
    ``ln(1 − e^{−S}) = ln S − S/2 + S²/24 − …`` is used instead: its next
    term is below 10⁻¹⁸ there, so the two branches agree to rounding.
    """
    lS = np.asarray(lS, dtype=float)
    small = lS < -20.0
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        return np.where(small,
                        lS - 0.5 * np.exp(lS),
                        _log1mexp(np.exp(np.where(small, 0.0, lS))))


def _bb6_terms_k(ka, kb, th, de):
    """``(lp, lq, lS, lg)`` — the logs of p, q, S = (p^δ + q^δ)^{1/δ} and
    G = 1 − e^{−S} — from ``ka = ln(1 − u)``, ``kb = ln(1 − v)``."""
    lp = _bb6_logp(ka, th)
    lq = _bb6_logp(kb, th)
    lS = np.logaddexp(de * lp, de * lq) / de
    return lp, lq, lS, _bb6_log_g(lS)


def _bb6_minus_s(lS, lx):
    """``x − S = −exp(ln S + ln1mexp(ln S − ln x))`` for x ∈ {p, q}, S ≥ x.

    The cancellation-free form of the module docstring: exactly 0 when
    S = x (δ = 1 with the other term absent), exactly −S when S ≫ x, and no
    ``exp`` of a large positive number anywhere.

    ``ln S − ln x ≥ 0`` mathematically (the ℓ^δ-norm dominates each term),
    but the two logs are rounded separately, so the difference comes out
    *negative* by an ulp whenever the other term is negligible — at
    (u, v) = (1 − 10⁻⁶, 1 − 10⁻¹²), δ = 10, the neglected ``(q/p)^δ`` is
    10⁻⁶¹. ``ln1mexp`` of a negative argument is NaN, so the difference is
    floored at 0, which is the value it rounds to anyway.
    """
    with np.errstate(invalid='ignore'):
        return -np.exp(lS + _log1mexp(np.maximum(lS - lx, 0.0)))


def _bb6_logpdf_k(ka, kb, th, de):
    """``ln c(u, v)`` from ``ka = ln(1 − u)``, ``kb = ln(1 − v)``."""
    lp, lq, lS, lg = _bb6_terms_k(ka, kb, th, de)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        # p + q − S, bracketed on the smaller of p, q (module docstring).
        excess = np.where(lp >= lq,
                          _bb6_minus_s(lS, lp) + np.exp(lq),
                          np.exp(lp) + _bb6_minus_s(lS, lq))
        # ln B, B = S(1 − 1/θ) + S·G/θ + (δ − 1)·G — three non-negative terms,
        # the first −∞ at θ = 1 and the third −∞ at δ = 1, absorbed exactly.
        log_th = math.log(th)
        log_b = np.logaddexp(
            np.logaddexp(_log_or_ninf(th - 1.0) - log_th + lS, lS + lg - log_th),
            _log_or_ninf(de - 1.0) + lg,
        )
        return (log_th
                + (de - 1.0) * (lp + lq)
                + (th - 1.0) * (ka + kb)
                + excess
                + (1.0 / th - 2.0) * lg
                + (1.0 - 2.0 * de) * lS
                + log_b)


def _bb6_cdf_k(ka, kb, th, de):
    """``(C, 1 − C)`` with ``1 − C = G^{1/θ} = exp(ln G / θ)``."""
    e = _bb6_terms_k(ka, kb, th, de)[3] / th
    return -np.expm1(e), np.exp(e)


def _bb6_logh_k(kb, ka, th, de):
    """``ln h(v|u)`` from ``kb = ln(1 − v)``, ``ka = ln(1 − u)``.

    ``ln h = (1/θ − 1) ln G + (1 − δ) ln S + (δ − 1) ln p + (θ − 1) ka +
    (p − S)``, the last term cancellation-free (:func:`_bb6_minus_s`); the
    textbook ``− ln m = p`` of ``ū^{θ−1}/m`` is already in it.
    """
    lp, _, lS, lg = _bb6_terms_k(ka, kb, th, de)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        return ((1.0 / th - 1.0) * lg
                + (1.0 - de) * lS
                + (de - 1.0) * lp
                + (th - 1.0) * ka
                + _bb6_minus_s(lS, lp))


def _bb6_h_k(kb, ka, th, de):
    """``(h, 1 − h)`` of h(v|u), through :func:`_bb6_logh_k`."""
    logh = _bb6_logh_k(kb, ka, th, de)
    return np.exp(logh), -np.expm1(logh)


def _bb6_inv_h_k(lw, ka, th, de):
    """``(v, 1 − v)`` solving h(v|u) = w, from ``lw = ln w``, ``ka = ln(1 − u)``.

    Vectorised bisection on ``x = logit v`` (module docstring): ln h is
    increasing in v hence in x, and ``kb = ln(1 − v) = −logaddexp(0, x)`` is
    exact, so 1 − v is never formed in linear scale. ``w`` outside
    ``[h(ε|u), h(1 − ε|u)]`` saturates to the nearer end, as
    :meth:`CopulaVirt.inv_h` does.
    """
    lw = np.asarray(lw, dtype=float)
    ka = np.asarray(ka, dtype=float)
    shape = np.broadcast(lw, ka).shape
    lw = np.broadcast_to(lw, shape)
    ka = np.broadcast_to(ka, shape)
    x_lo = math.log(EPS) - math.log1p(-EPS)
    lo = np.full(shape, x_lo)
    hi = np.full(shape, -x_lo)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        below = lw <= _bb6_logh_k(math.log1p(-EPS), ka, th, de)
        above = lw >= _bb6_logh_k(math.log(EPS), ka, th, de)
        for _ in range(_BISECT_STEPS):
            mid = 0.5 * (lo + hi)
            kb = -np.logaddexp(0.0, mid)            # ln(1 − expit(mid))
            up = _bb6_logh_k(kb, ka, th, de) < lw
            lo = np.where(up, mid, lo)
            hi = np.where(up, hi, mid)
        x = 0.5 * (lo + hi)
        x = np.where(below, x_lo, np.where(above, -x_lo, x))
        kb = -np.logaddexp(0.0, x)
    return -np.expm1(kb), np.exp(kb)


# ---------------------------------------------------------------------------
# Copula class
# ---------------------------------------------------------------------------

class CopulaBB6(CopulaVirt):
    """BB6 (Joe-Gumbel) copula — upper-tail dependence only, two parameters.

    Parameters
    ----------
    tau_k : float
        Kendall's τ ∈ (0, 1). Must satisfy τ ≥ 1 − 1/δ so that
        τ_Joe = 1 − δ(1 − τ) ∈ [0, 1) has a Joe parameter θ ≥ 1.
    delta6 : float, optional
        Outer-power exponent δ ≥ 1 (default 1.0, the Joe member).
        δ = 1 is Joe; δ = 1/(1 − τ) is Gumbel–Hougaard (θ = 1); at fixed τ a
        larger δ moves dependence from the "Joe" shape towards the Gumbel
        one. Named ``delta6``, not ``delta``, so that the joint fitter cannot
        route BB6 through BB1's branch (module docstring).
    """

    n_params: int = 2

    def __init__(self, **kwargs):
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    # ------------------------------------------------------------------
    # Parameters — the joint (τ, δ) constraint (module docstring)
    # ------------------------------------------------------------------

    @classmethod
    def delta_max(cls, tau: float) -> float:
        """Largest δ admissible at this τ: the biggest **double** with
        ``δ·(1 − τ) ≤ 1``, i.e. τ_Joe = 1 − δ(1 − τ) ≥ 0.

        ``1/(1 − τ)`` is correctly rounded, so it can be one ulp above the
        largest admissible double (τ = 0.9: the product is 1 + 2·10⁻¹⁶) or one
        ulp below it (τ = 1/3: ``1/(1 − τ) = 1.4999999999999998`` while
        δ = 1.5 still multiplies to exactly 1.0). It is therefore walked down
        while the product the *constructor* computes exceeds 1, then up while
        the next double still passes — the hook and the constructor then agree
        to the last bit. Unlike BB1's ``delta_max``, the value is *attained*:
        δ = 1/(1 − τ) is θ = 1, the Gumbel member, not a degenerate limit.
        """
        gap = 1.0 - float(tau)
        if not gap > 0.0:
            return 1.0
        hi = 1.0 / gap
        for _ in range(4):                       # down: make the product ≤ 1
            if hi <= 1.0 or hi * gap <= 1.0:
                break
            hi = float(np.nextafter(hi, 1.0))
        for _ in range(4):                       # up: take the last double that passes
            nxt = float(np.nextafter(hi, math.inf))
            if nxt * gap > 1.0:
                break
            hi = nxt
        return max(1.0, float(hi))

    @classmethod
    def constrain_params(cls, params: dict) -> dict:
        """Clamp δ into ``[1, delta_max(τ)]`` — a box cannot express it.

        The projection a bounded optimiser needs (BB1's precedent); the joint
        MLE does not rely on it, because ``_two_parameter_spec``'s ``delta6``
        branch optimises in (θ, δ), whose box *is* the admissible set.
        """
        out = dict(params)
        tau = float(out.get("tau_k", 0.0))
        delta = float(out.get("delta6", _DELTA_DEFAULT))
        out["delta6"] = float(np.clip(delta, 1.0, max(1.0, cls.delta_max(tau))))
        return out

    @classmethod
    def constructible_params(cls, params: dict) -> dict:
        """``params`` itself when BB6 builds at it; else δ moved into ``[1, δ_max(τ)]``.

        The test is the constructor's own (:meth:`_update_params`, the same
        floating-point operations). Only a refused pair changes, and only δ —
        τ is the value the caller drew (multistart jitter, family draw) and
        stays. Both ends of the interval are admissible here, so the repaired
        δ goes **on** the end, without BB1's ``TAU_PAD_REL`` pull-in: BB1
        pads because its upper end θ = 0 is refused, while BB6's is θ = 1,
        the Gumbel member. A pair no δ repairs (τ ≥ 1, τ ≤ 0, non-finite) is
        returned unchanged, for the constructor to refuse.
        """
        tau = params.get("tau_k")
        delta = params.get("delta6", _DELTA_DEFAULT)
        try:
            tau_f, delta_f = float(tau), float(delta)
        except (TypeError, ValueError):
            return params
        if isinstance(delta, (bool, np.bool_)):     # the constructor refuses it
            return params
        if cls._delta_error(tau_f, delta_f) is None:
            return params
        if not (np.isfinite(tau_f) and 0.0 < tau_f < 1.0):
            return params
        return {**params, "delta6": min(max(delta_f, 1.0), cls.delta_max(tau_f))}

    @classmethod
    def _delta_error(cls, tau: float, delta: float) -> str | None:
        """Why the constructor refuses (τ, δ), or ``None`` when it accepts."""
        if not np.isfinite(delta) or delta < 1.0:
            return f"BB6: delta6 must be ≥ 1, got {delta!r}."
        gap = 1.0 - float(tau)
        if not gap > 0.0:
            return f"BB6: tau_k={tau!r} must be < 1."
        if delta * gap > 1.0:
            return (f"BB6: tau_k={tau!r} is not reachable with delta6={delta!r} — "
                    f"BB6 needs tau >= 1 - 1/delta6 = {1.0 - 1.0 / delta:.6g} "
                    f"(delta6 <= 1/(1 - tau) = {1.0 / gap:.6g}), so that the Joe "
                    f"parameter theta = theta_Joe(1 - delta6*(1 - tau)) exists.")
        return None

    def _update_params(self):
        tau = float(self.params['tau_k'])
        delta = self.params.get('delta6', _DELTA_DEFAULT)
        if isinstance(delta, (bool, np.bool_)):
            raise CopulaParameterError(f'BB6: delta6={delta!r} is not a number.')
        try:
            delta = float(delta)
        except (TypeError, ValueError):
            raise CopulaParameterError(f'BB6: delta6={delta!r} is not a number.') from None
        err = self._delta_error(tau, delta)
        if err is not None:
            raise CopulaParameterError(err)
        # (★) τ = 1 − (1 − τ_Joe(θ))/δ inverted: τ_Joe ∈ [0, 1) by the test
        # above. At δ = 1 the identity is τ_Joe = τ, used directly rather than
        # as ``1 − 1·(1 − τ)`` (which loses the last bits: 1 − (1 − 0.05) =
        # 0.050000000000000044), so the Joe member builds the very θ
        # ``CopulaJoe`` builds — bit for bit, as Tawn's ψ = 1 member does for
        # ``CopulaGH``. Below ``_TAU_JOE_FLOOR`` no θ > 1 is representable.
        tau_joe = tau if delta == 1.0 else 1.0 - delta * (1.0 - tau)
        self.theta = _joe_theta_from_tau(0.0 if tau_joe < _TAU_JOE_FLOOR else tau_joe)
        self.delta6 = delta
        self.params['delta6'] = delta

    def tau_of(self) -> float:
        """τ recomputed from (θ, δ) by (★) — the forward map of the constructor."""
        return 1.0 - (1.0 - _joe_tau_from_theta(self.theta)) / self.delta6

    # ------------------------------------------------------------------
    # Kernel interface — Joe's coordinates (module docstring)
    # ------------------------------------------------------------------

    @staticmethod
    def _kcoord(x):
        return np.log1p(-x)

    @staticmethod
    def _kcoord_reflected(x):
        return np.log(x)

    def _k_logpdf(self, ka, kb):
        return _bb6_logpdf_k(ka, kb, self.theta, self.delta6)

    def _k_cdf(self, ka, kb):
        return _bb6_cdf_k(ka, kb, self.theta, self.delta6)

    def _k_h(self, kb, ka):
        return _bb6_h_k(kb, ka, self.theta, self.delta6)

    def _k_inv_h(self, lw, ka):
        return _bb6_inv_h_k(lw, ka, self.theta, self.delta6)

    # ------------------------------------------------------------------
    # CDF / PDF / h-function
    # ------------------------------------------------------------------

    def cdf(self, uv):
        """C(u,v) = 1 − G^{1/θ}, G = 1 − exp(−S) (module docstring)."""
        u = np.float64(minmaxEPS(uv[0]))
        v = np.float64(minmaxEPS(uv[1]))
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            c, _ = self._k_cdf(self._kcoord(u), self._kcoord(v))
        return float(np.clip(c, 0.0, 1.0))

    def cdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised closed-form CDF."""
        uv = np.asarray(uv, dtype=float)
        u = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            c, _ = self._k_cdf(self._kcoord(u), self._kcoord(v))
        return np.clip(c, 0.0, 1.0)

    def pdf(self, uv):
        """c(u,v) = θ (pq)^{δ−1} (ū v̄)^{θ−1} e^{p+q−S} G^{1/θ−2} S^{1−2δ} B.

        Evaluated as ``exp`` of the log-space kernel; 0.0 only on underflow.
        """
        u = np.float64(minmaxEPS(uv[0]))
        v = np.float64(minmaxEPS(uv[1]))
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            return float(np.exp(self._k_logpdf(self._kcoord(u), self._kcoord(v))))

    def pdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised closed-form PDF, ``exp`` of :meth:`logpdf_array` (no floor)."""
        return np.exp(self.logpdf_array(uv))

    def logpdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Native log-PDF, computed entirely in log space (module docstring)."""
        uv = np.asarray(uv, dtype=float)
        u = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            return self._k_logpdf(self._kcoord(u), self._kcoord(v))

    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = G^{1/θ−1} e^{−S} S^{1−δ} p^{δ−1} ū^{θ−1} / m."""
        u = np.float64(minmaxEPS(u))
        v = np.float64(minmaxEPS(v))
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            h, _ = self._k_h(self._kcoord(v), self._kcoord(u))
        return float(np.clip(h, 0.0, 1.0))

    def inv_h(self, w: float, u: float) -> float:
        """v such that h(v|u) = w, by the bisection of the module docstring."""
        return float(self.inv_h_array(np.array([float(w)]), np.array([float(u)]))[0])

    def inv_h_array(self, w: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Vectorised inverse h-function (module docstring)."""
        w = np.clip(np.asarray(w, dtype=float), EPS, ONE_MINUS_EPS)
        u = np.clip(np.asarray(u, dtype=float), EPS, ONE_MINUS_EPS)
        if w.shape != u.shape:
            raise ValueError(f"inv_h_array: w {w.shape} and u {u.shape} must match.")
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            v, _ = self._k_inv_h(np.log(w), self._kcoord(u))
        return np.clip(v, EPS, ONE_MINUS_EPS)

    def tail_dependence(self) -> tuple[float, float]:
        """λ_L = 0, λ_U = 2 − 2^{1/(θδ)} (module docstring)."""
        return 0.0, float(2.0 - 2.0 ** (1.0 / (self.theta * self.delta6)))

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    @classmethod
    def fit(cls, data: np.ndarray, method: str = 'mle'):
        """Joint MLE of (τ, δ); ``method='tau'`` falls back to it.

        Kendall's τ alone cannot identify δ — (★) has two unknowns — so a
        ``'tau'`` request logs a warning and is answered by the same joint
        two-parameter MLE as ``'mle'`` (BB1's and Tawn's precedent).
        """
        if method == 'tau':
            logger.warning(
                "%s.fit: method='tau' cannot identify delta6 from Kendall's tau "
                "alone; falling back to MLE.", cls.__name__)
            method = 'mle'
        return super().fit(data, method=method)


if __name__ == '__main__':
    from pathlib import Path

    from pmcprg.copulas.archimedean.gumbel import CopulaGH
    from pmcprg.copulas.archimedean.joe import CopulaJoe

    cop = CopulaBB6(tau_k=0.6, delta6=1.5)
    lL, lU = cop.tail_dependence()
    print(f'Copula : {cop.copula_enum.value.LONG_NAME}')
    print(f'tau_k  : {cop.params["tau_k"]:.4f}  delta6={cop.delta6:.4f}  theta={cop.theta:.6f}')
    print(f'tau    check (★): {cop.tau_of():.15f}  (want {cop.params["tau_k"]:.15f})')
    print(f'pdf(0.3, 0.7) = {cop.pdf([0.3, 0.7]):.6f}')
    print(f'cdf(0.3, 0.7) = {cop.cdf([0.3, 0.7]):.6f}')
    print(f'h(0.7 | 0.3)  = {cop.conditional_cdf(0.7, 0.3):.6f}')
    print(f'tail dep : λ_L = {lL:.4f},  λ_U = {lU:.4f}  [= 2 − 2^(1/(θδ))]')

    print('\nδ = 1 vs Joe @ (0.3, 0.7), τ = 0.6:')
    print(f'  BB6.pdf={CopulaBB6(tau_k=0.6, delta6=1.0).pdf([0.3, 0.7]):.12f}  '
          f'Joe.pdf={CopulaJoe(tau_k=0.6).pdf([0.3, 0.7]):.12f}')
    print('θ = 1 vs Gumbel @ (0.3, 0.7), τ = 0.5, δ = 2:')
    print(f'  BB6.pdf={CopulaBB6(tau_k=0.5, delta6=2.0).pdf([0.3, 0.7]):.12f}  '
          f'GH.pdf={CopulaGH(tau_k=0.5).pdf([0.3, 0.7]):.12f}')

    plot_dir = Path('./data/Plots/Copulas')
    plot_dir.mkdir(parents=True, exist_ok=True)
    cop.plot_pdf(plot_dir)
    cop.plot_cdf(plot_dir)
    cop.plot_h_function(plot_dir)
    cop.plot_samples(plot_dir)
    cop.plot_overview(plot_dir)
