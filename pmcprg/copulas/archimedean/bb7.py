"""
BB7 copula (Joe-Clayton) — both tails, two parameters (FR-9).

Generator and CDF (re-derived, not transcribed)
-----------------------------------------------
The generator VineCopula and vinecopulib use for BB7 is

    φ(t) = [ 1 − (1 − t)^θ ]^{−δ} − 1 ,      θ ≥ 1,  δ > 0,

and this round did **not** take that on trust. What could be confirmed here,
and how:

* φ(1) = 0, φ(0⁺) = +∞ and φ′ < 0 on (0, 1) by inspection, so φ is a *strict*
  Archimedean generator;
* ``C = φ^{-1}(φ(u) + φ(v))`` evaluated at 120 digits in ``mpmath`` has
  uniform margins (``C(u, 1) = u`` to 10⁻³⁰) and is 2-increasing (no negative
  rectangle volume on an 11 × 11 grid, at sixteen (θ, δ)) — i.e. the candidate
  *is* a copula, not merely a plausible formula;
* the closed form below reproduces ``φ^{-1}(φ(u) + φ(v))`` to 3·10⁻²²
  relative (the limit of the 120-digit round trip, not of the identity);
* its two sub-models are Clayton and Joe **exactly** (see below), which is
  what "Joe-Clayton" has to mean, and its tail coefficients come out as
  λ_U = 2 − 2^{1/θ}, λ_L = 2^{−1/δ} — precisely the pair Patton (2006)
  reparametrises BB7 by, under the name *symmetrised Joe-Clayton*. Only this
  generator produces that pair.
* What could **not** be done here is to read Joe (1997) ch. 5 itself; the
  confirmation above is numerical and internal, not bibliographic.

Inverting φ — ``s = φ(t)`` ⇒ ``(1 + s)^{−1/δ} = 1 − (1 − t)^θ`` — gives

    φ^{-1}(s) = 1 − [ 1 − (1 + s)^{−1/δ} ]^{1/θ},

and with ū = 1 − u, v̄ = 1 − v, a = 1 − ū^θ, b = 1 − v̄^θ,

    T = a^{−δ} + b^{−δ} − 1 ≥ 1,   R = T^{−1/δ} ∈ (0, 1],   K = 1 − R,

the Archimedean construction ``C = φ^{-1}(φ(u) + φ(v))`` reduces to

    C(u, v) = 1 − K^{1/θ} = 1 − [ 1 − (a^{−δ} + b^{−δ} − 1)^{−1/δ} ]^{1/θ}.

The **complement** ū appears inside the generator — as in Joe's and BB6's,
not BB1's — so the natural kernel coordinate is ``ka = ln(1 − u)``.

Sub-models (verified, both exact)
---------------------------------
* **θ = 1** → ``φ(t) = t^{−δ} − 1``, which *is* **Clayton's** generator at
  parameter δ: ``a = u``, ``T = u^{−δ} + v^{−δ} − 1``, ``K = 1 − R`` and
  ``C = 1 − K = R = T^{−1/δ}``, Clayton's own CDF. Checked at 120 digits:
  max |ΔC| = 4.8·10⁻¹²².  τ(1, δ) = δ/(δ + 2) is Clayton's τ, algebraically
  (below) and to the last of 20 digits.
* **δ → 0⁺** → ``φ(t)/δ = [e^{−δ ln a} − 1]/δ → −ln a = −ln(1 − (1 − t)^θ)``,
  which is **Joe's** generator; an Archimedean generator is defined up to a
  positive factor, so the limit family is Joe at the same θ. Checked at 120
  digits: |ΔC| falls exactly linearly in δ (7.9·10⁻¹⁰ at δ = 10⁻⁸,
  7.9·10⁻²² at δ = 10⁻²⁰), and τ(θ, 10⁻²⁵) = τ_Joe(θ) to all 20 digits
  printed. This makes BB7 a genuine **two-way** nest of Clayton and Joe —
  unlike BB6, which nests Joe and Gumbel–Hougaard.
* Independence is the corner of the two: θ = 1 *and* δ → 0.

θ = 1 is a *boundary* of the admissible set and δ = 0 is the open end of it,
which is what makes both likelihood-ratio tests one-sided
(``_stderr._SUBMODEL_NESTING``). In the coordinates this module actually
stores (θ, below), that is the **lower** end θ = 1 for Clayton and the
**upper** end θ → θ_Joe(τ) for Joe. Note the asymmetry with BB6: BB6's Joe
member is the attained δ = 1, BB7's is a limit no parameter realises.

Kendall's τ — a Beta closed form with a removable singularity
-------------------------------------------------------------
BB6's outer-power identity does **not** apply: BB7's generator is not
``ψ(t)^δ`` for a fixed base ψ. Genest & MacKay's ``τ = 1 + 4∫₀¹ φ/φ′`` is
therefore integrated from scratch. With φ′ = −δθ ū^{θ−1} a^{−δ−1},

    φ/φ′ = −(1/(δθ)) · (a − a^{δ+1}) / ū^{θ−1},

and the substitution ū = 1 − t, then x = ū^θ (so ``a = 1 − x``) turns the
integral into a difference of Beta functions,

    τ = 1 − (4/(δθ²)) [ B(κ, 2) − B(κ, δ + 2) ],      κ = 2/θ − 1.

Each Beta diverges for θ ≥ 2 while the difference does not, so this is not
the form to evaluate. Using ``B(κ, 2) = 1/(κ(κ+1))`` and
``B(κ, δ+2) = (δ+1)B(κ+1, δ+1)/κ`` — both plain Γ-algebra — the whole 1/κ
factors out and what is left is a single ordinary Beta:

    τ(θ, δ) = 1 − 2 [ 1 − (2/θ)(δ + 1) B(2/θ, δ + 1) ] / ( δ (2 − θ) )   (★)

with ``2/θ ∈ (0, 2]`` and ``δ + 1 > 1``: B is finite on the whole admissible
set. **The removable singularity is θ = 2**, where ``2 − θ = 0`` and the
bracket vanishes too — ``(2/θ)(δ+1)B(2/θ, δ+1) = 1`` at θ = 2 for every δ —
a genuine 0/0. Its limit, by one derivative of the bracket in κ, is

    τ(2, δ) = 1 − [ ψ(δ + 2) − ψ(2) ] / δ                                (★₂)

(the *same* θ = 2 at which Joe's own ``1 + 2[ψ(2) − ψ(1 + 2/θ)]/(2 − θ)`` is
0/0 — ``joe.py``'s middle branch — which is no coincidence: δ → 0 of (★₂)
is 1 − ψ′(2) = 2 − π²/6, Joe's τ at θ = 2). A second, weaker 0/0 sits at
δ → 0, where the bracket vanishes as well and the limit is τ_Joe(θ).

Both are removed at once by writing the bracket as ``1 − e^L`` with

    L(δ, κ) = lnΓ(2 + δ) + lnΓ(2 + κ) − lnΓ(2 + δ + κ),     κ = 2/θ − 1,

(``e^L = (2/θ)(δ+1)B(2/θ, δ+1)``, pure Γ-algebra again) and noting that L is
**symmetric** in (δ, κ) and vanishes when *either* argument does. Since
``δ(2 − θ) = δθκ``,

    τ = 1 − (2/θ) · W(δ, κ) · (1 − e^L)/L ,       W = L/(δκ),            (★★)

in which ``(1 − e^L)/L → −1`` as L → 0 and W is analytic at δ = 0 and at
κ = 0. W itself never forms L as a difference of logΓ's: differentiating L
once in each argument gives ``∂²L/∂δ∂κ = −ψ′(2 + δ + κ)``, hence the
**positive-term** expansions (no cancellation anywhere)

    W = −(1/g) [ ψ(2 + g) − ψ(2)
                 + Σ_{j≥1} (−1)^{j+1} sʲ ( ζ(j+1, 2+g) − ζ(j+1, 2) )/(j+1) ]

for (g, s) = (δ, κ) or, by the symmetry of L, (κ, δ) — the series in s has
radius 2 in κ and 1 + 2/θ > 1 in δ, so ``|s| ≤ 1/2`` always halves per term.
δ ≥ 1/2 takes the first form, |κ| ≥ 1/2 the second, and when *both* are
below 1/2 the double series

    W = −Σ_{m≥1} Σ_{i=1}^{m} (−1)^{m+1} (ζ(m+1) − 1) C(m, i)/(m − i + 1)
                             · κ^{m−i} δ^{i−1}

is used, which is regular in both. Measured against a 120-digit ``mpmath``
evaluation of (★) on θ ∈ [1, 10¹²] × δ ∈ [10⁻¹², 10]: **|Δτ| ≤ 1.2·10⁻¹⁵
absolute** everywhere, ≤ 5.6·10⁻¹⁰ *relative* for τ ≥ 10⁻⁶, and
≤ 2.5·10⁻¹⁵ relative on 1 − τ. The relative accuracy of τ itself does
degrade in the doubly-degenerate corner θ → 1 **and** δ → 0, where τ → 0 and
(★★) cancels: 7·10⁻⁵ relative at τ = 2·10⁻¹² (still 1.4·10⁻¹⁶ absolute).
This is weaker than ``joe.py``'s τ, which is exact from θ = 1 (audit RB-2);
no factored form of (★★) in (θ − 1, δ) was found, and the corner is only
reachable where the joint constraint τ > τ_Joe(θ) has already squeezed θ to
1 and the recovered δ to the order of τ. Stated rather than hidden.

And the whole of (★)/(★★) was checked against an *independent* 120-digit
Genest–MacKay quadrature ``1 + 4∫φ/φ′`` of BB7's own generator (φ′ itself
first validated against ``mpmath``'s numerical differentiation, 3.5·10⁻¹²⁰):
relative agreement 9·10⁻⁶¹ to 4·10⁻⁵⁸ — the quadrature's limit, not (★)'s.

Parametrisation: θ is the extra parameter, **not** δ
----------------------------------------------------
The package stores ``tau_k`` plus the extra parameters, so one of (θ, δ) is
the extra one and the other is recovered. BB1 and BB6 both keep δ; **BB7
cannot**, and the reason is a property of the family worth stating plainly
rather than inheriting from its neighbours:

    **τ is *not* monotone in θ at fixed δ.**

At fixed δ, τ starts at Clayton's ``τ(1, δ) = δ/(δ + 2)``, *dips*, and only
then rises to 1. The dip appears at

    δ* ≈ 3.44121781421      (the δ where ∂τ/∂θ|_{θ=1} changes sign,
                             located by bisection at 40 digits)

and deepens: at δ = 5 the minimum is 0.00823 below the Clayton value, at
δ = 10 it is 0.0389 (at θ = 3), at δ = 20 it is 0.0595. Keeping δ as the
extra parameter would therefore make (τ, δ) **many-to-one** — two θ for every
τ in the dip — and, worse, would hide a whole slab of genuine BB7 copulas
that no (τ, δ) pair can name. It would also hand the joint MLE the flat
plateau audit RB-8 is about. This was *measured*, not assumed: an earlier
draft of this module did keep δ, on the analogy with BB1 and BB6, and its own
monotonicity test is what caught it.

τ **is** strictly increasing in δ at fixed θ (checked at 50 digits over
θ ∈ [1, 10⁵] × δ ∈ [10⁻⁸, 10⁴]), from ``τ_Joe(θ)`` as δ → 0⁺ to 1 as δ → ∞.
So ``theta7`` is the extra parameter, δ is recovered by Brent on ``ln δ``
(:func:`_bb7_delta_from_tau`, unique root), and the joint constraint is

    τ > τ_Joe(θ)      ⟺      θ < θ_Joe(τ),

with ``θ = 1`` — the Clayton member, δ = 2τ/(1 − τ) in closed form — at the
**lower** end and the Joe limit δ → 0 at the upper one. Both ends are
admissible *doubles*: θ = 1 is a genuine member, and a θ one ulp below
``θ_Joe(τ)`` is a BB7 with a very small δ, so :meth:`CopulaBB7.theta_max`
returns the largest double passing the constructor's own strict test and
:meth:`constructible_params` repairs a refused draw **on** it, without BB1's
``TAU_PAD_REL`` pull-in. Unlike BB6's δ = 1 but exactly like it in spirit,
**θ = 1 is admissible at every registered τ**, so a block without ``theta7``
(:data:`_THETA_DEFAULT`) always builds — the Clayton member.

A δ the Brent bracket cannot separate from 0 is answered with
:data:`_DELTA_FLOOR` = 10⁻¹⁴, which is Joe to machine precision
(λ_L = 2^{−10¹⁴} = 0.0) and moves τ by less than 10⁻¹⁴; that band is reached
routinely, because ``constructible_params`` repairs onto ``theta_max(τ)``
(BB6's ``_TAU_JOE_FLOOR`` situation, in BB7's coordinates).

*Why the name ``theta7`` and not ``delta``/``delta6``.*
``_fit._two_parameter_spec`` dispatches on the **parameter name**:
``"delta"`` is BB1's branch, with BB1's ``τ = 1 − 2/(δ(θ + 2))`` map, and
``"delta6"`` is BB6's. BB7 routed through either would be fitted with the
wrong parametrisation — silently, since every pair is numerically plausible.
Same reason as t-EV's ``nu`` versus Student's ``df``. ``test_bb7.py`` proves
BB7 takes neither path.

Tail dependence
---------------
From ``1 − φ^{-1}(s) = [1 − (1 + s)^{−1/δ}]^{1/θ} ≈ (s/δ)^{1/θ}`` as s → 0,
``λ_U = 2 − lim (1 − φ^{-1}(2s))/(1 − φ^{-1}(s))``; and from
``φ^{-1}(s) ≈ (1 + s)^{−1/δ}/θ`` as s → ∞, ``λ_L = lim φ^{-1}(2s)/φ^{-1}(s)``:

    λ_U = 2 − 2^{1/θ},              λ_L = 2^{−1/δ},

each depending on **one** parameter only — the analytic expression of the
two-way nest, and the reason Patton (2006) uses (λ_U, λ_L) as the
parametrisation. Checked at 120 digits against the diagonal ratios
``(1 − 2u + C(u,u))/(1 − u)`` at u = 1 − 10⁻²⁵ and ``C(u,u)/u`` at u = 10⁻²⁵:
agreement to 10 or more digits at every (θ, δ) tried. λ_L → 0 as δ → 0 (Joe)
and λ_U = 0 at θ = 1 (Clayton), as they must.

Densities (derivation)
----------------------
Only a depends on u and only b on v. With ``∂a/∂u = θ ū^{θ−1}``,
``∂T/∂u = −δθ ū^{θ−1} a^{−δ−1}`` and ``C = 1 − K^{1/θ}``:

    h(v|u) = ∂C/∂u = K^{1/θ−1} ū^{θ−1} a^{−δ−1} T^{−1/δ−1},

    c(u, v) = (ū v̄)^{θ−1} (ab)^{−δ−1} K^{1/θ−2} T^{−1/δ−2} · B,
    B = (θ − 1)·R + θ(1 + δ)·K.

Both terms of B are ≥ 0 **because** θ ≥ 1 and δ > 0. At θ = 1 the first
vanishes and the pair collapses to Clayton's ``(1 + δ)(uv)^{−δ−1}
T^{−1/δ−2}`` and ``u^{−δ−1} T^{−1/δ−1}``; as δ → 0 they collapse to Joe's.
Both closed forms were checked against ``mpmath``'s own first and mixed
derivatives of C: max relative error 9.4·10⁻¹¹⁸ (h) and 2.1·10⁻¹¹⁷ (c).

Numerics
--------
Everything is evaluated from ``ka = ln(1 − u)``, ``kb = ln(1 − v)`` and
nothing is formed in linear scale. ``ln(−ln a)`` is BB6's ``_bb6_logp``,
imported rather than copied (a third copy of Mächler's ``log1mexp`` switch,
and of the ``z > 700`` asymptote that keeps ``ln(−ln a) = −z`` once ū^θ
underflows, would be a place for the two families to drift). From it:

* ``ln a = −e^{lp}`` — always finite, |ln a| ≤ 37 on the clipped square;
* ``ln(a^{−δ} − 1) = lx + ln1mexp(lx)`` with ``lx = δ e^{lp}``, except below
  ``lp = −700`` where ``lx`` underflows and the exact value is ``ln δ + lp``
  to a relative 10⁻³⁰⁴. Without that branch ``logpdf_array`` is −∞ on the
  whole ``u = 1 − 10⁻¹²`` edge as soon as θ ≳ 20;
* ``ln(T − 1) = logaddexp`` of the two, then ``ln T = logaddexp(0, ln(T−1))``
  — T is never formed as ``a^{−δ} + b^{−δ} − 1``, whose two terms cancel to
  nothing precisely where the density is largest (u, v → 1);
* ``ln K = ln(1 − e^{−lnT/δ}) = ln1mexp(lnT/δ)``, again with the
  ``ln(T−1) < −700`` asymptote ``ln K = ln(T−1) − ln δ`` for the range where
  ``ln T`` itself underflows to 0.

``ln B`` is a ``logaddexp`` of its two ≥ 0 terms, so ``ln(θ − 1) = −∞`` at
θ = 1 is absorbed exactly rather than cancelled. Measured against the
adaptive-precision ``mpmath`` ground truth of ``test_bb7.py`` (120 to 3840
digits, C from the generator alone, c and h its finite differences) on the
8 × 8 grid u, v ∈ [10⁻¹², 1 − 10⁻¹²] at eight (τ, θ) from (0.2, 1) to
(0.9, 3) — recovered δ from 0.4 to 45 — both sub-model ends included:
**≤ 2.4·10⁻¹⁴ on ln c** (absolute, and relative where |ln c| > 1),
**≤ 1.0·10⁻¹⁴ relative on C**, **≤ 2.3·10⁻¹³ relative on h**. (The reference
itself had to be told how much precision to start at, twice: ``1 − ū^θ`` is
``1 − 10⁻¹¹⁸⁸`` at θ = 198 and u = 1 − 10⁻⁶ — a point an earlier draft of
this module's parametrisation reached — which rounds to exactly 1, and C to
exactly 1, silently, below ≈ 1200 digits; and a difference quotient of
10⁻²⁴² comes out as exactly 0 below ≈ 280, which is the corner
(τ = 0.9, θ = 3) still hits. Both "converged" at 240 digits on a wrong value
until the ladder was told otherwise. A reference is a measurement too; BB6's
round learned the same lesson about its quadrature mesh.) There is no
fallback — a value that cannot be computed is
NaN, not ``EPS`` (audit RB-9).

*Kernel interface* (audit FR-8's shape). ``_kcoord(x) = ln(1 − x)`` and
``_kcoord_reflected(x) = ln x``, Joe's and BB6's, because BB7's generator is
built from the complement 1 − t. BB7 registers no rotation or survival
wrapper of its own — FR-8 is closed, and its fixed list of six families does
not include the BB pairs — but the interface costs nothing and is the shape
such a wrapper would consume.

*Inverse h-function.* No closed form (as for BB1, BB6 and Tawn): K depends on
v through T, so ``ln h(v|u) = ln w`` is solved by 68 steps of vectorised
bisection on ``logit v`` over ``[logit ε, logit(1 − ε)]`` (final bracket
< 10⁻¹⁹ in logit), which never forms 1 − v in linear scale —
``ln(1 − v) = −logaddexp(0, logit v)`` — with :meth:`CopulaVirt.inv_h`'s
saturation convention at the two ends.

Fitting
-------
``fit`` is the joint two-parameter MLE by default
(:func:`pmcprg.copulas._fit._fit_two_parameter_mle`); τ alone cannot identify
θ, so ``method='tau'`` is itau — τ̂, then θ by maximum likelihood at that τ
(FR-12; it used to warn and fall back to MLE). ``_two_parameter_spec``'s ``theta7`` branch optimises in
``(ln(θ − 1), ln δ)``, whose box **is** the admissible set — every θ ≥ 1,
δ > 0 is a BB7 copula — with τ mapped back by (★★) and θ carried through
unchanged. δ is the *nuisance* coordinate of the fit, not a stored parameter,
for the monotonicity reason above.

Known gaps for the roadmap (deliberate, not oversights)
-------------------------------------------------------
* No 90°/270° rotation is registered: FR-8 closed on a fixed list of six
  families that does not include BB6/BB7/BB8. BB7 reaches τ > 0 only.
* BB7 is not in ``_stderr._spec_of``'s Oakes / Lystig–Hughes analytic
  standard-error machinery, which FR-4 explicitly defers for every two- and
  three-parameter family beyond BB1 and Student.
* ``mle_tau_discrepancy_test`` and ``dpd_fit`` (FR-7) are **one-parameter**
  estimators: they refuse every two-parameter family — BB1, BB6, Student and
  now BB7 — with a documented ``NotImplementedError``, because Kendall's τ
  does not identify a second parameter. ``test_bb7.py`` pins that BB7 gets
  that reasoned refusal (the message names its own ``theta7``) rather than a
  crash; lifting it is a separate FR-7 job.

References
----------
* Joe, H. (1993). Parametric families of multivariate distributions with
  given margins. *Journal of Multivariate Analysis* 46(2), 262–282,
  doi:10.1006/jmva.1993.1061 (the BB construction; **note the DOI**: the
  commonly copied ``10.1006/jmva.93.1061`` does not resolve — checked
  against Crossref, which returns this article for ``jmva.1993.1061``).
* Joe, H. (1997). *Multivariate Models and Dependence Concepts*, Chapman &
  Hall/CRC, ch. 5, doi:10.1201/b13150 (family BB7 and its sub-models).
* Patton, A. J. (2006). Modelling asymmetric exchange rate dependence.
  *International Economic Review* 47(2), 527–556,
  doi:10.1111/j.1468-2354.2006.00387.x (verified against Crossref; the
  "symmetrised Joe-Clayton" reparametrisation by (λ_U, λ_L), which only the
  generator above produces).
* Genest, C. & MacKay, J. (1986). The joy of copulas: bivariate
  distributions with uniform marginals. *The American Statistician* 40(4),
  280–283, doi:10.1080/00031305.1986.10475414 (τ = 1 + 4∫φ/φ′, integrated
  above to give (★)).
* Hofert, M., Mächler, M. & McNeil, A. J. (2012). Likelihood inference for
  Archimedean copulas in high dimensions under known margins.
  *J. Multivariate Anal.* 110, 133–150, doi:10.1016/j.jmva.2012.02.019
  (log-space evaluation of Archimedean densities).
* Mächler, M. (2012). *Accurately computing log(1 − exp(−|a|))*, vignette of
  the R package Rmpfr (the ln 2 switch, reused through ``joe.py``).
* Schepsmeier, U. et al. (2018). *VineCopula*, R package (family 9, BB7,
  same (θ, δ) convention).
"""
if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import logging
import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import digamma, zeta

from pmcprg.copulas._base import CopulaVirt
from pmcprg.copulas.archimedean.bb6 import _Z_ASYMPTOTIC, _bb6_logp
from pmcprg.copulas.archimedean.joe import (
    _joe_tau_from_theta,
    _joe_theta_from_tau,
    _log1mexp,
)
from pmcprg.exceptions import CopulaParameterError
from pmcprg.numerics import EPS, ONE_MINUS_EPS, minmaxEPS

logger = logging.getLogger(__name__)

# A block without ``theta7`` is the **Clayton member** θ = 1, admissible at
# every registered τ (module docstring, "Parametrisation") — BB6's ``delta6``
# = 1 principle (the sub-model that is universally admissible), applied to the
# sub-model that happens to be universally admissible here.
_THETA_DEFAULT: float = 1.0

# Brent bracket for δ(τ, θ). The lower end is also the **Joe stand-in**: below
# it λ_L = 2^{−10¹⁴} is exactly 0.0 and τ is within 10⁻¹⁴ of τ_Joe(θ), so a τ
# that no representable δ separates from the Joe limit is answered with it —
# BB6's ``_TAU_JOE_FLOOR`` situation, and reached the same way, by
# ``constructible_params`` repairing a refused θ *onto* ``theta_max(τ)``.
# The upper end is generous rather than tight: at the box's θ = 10 and the
# padded τ ceiling 0.9999, δ ≈ 10¹⁷ (τ → 1 like 2/((θ−2)δ^{2/θ}) for θ > 2,
# so large θ needs a large δ), and nothing here forms Γ(2 + δ) itself.
_DELTA_FLOOR: float = 1e-14
_DELTA_CEIL: float = 1e150

# Bisection steps of the inverse h-function, on logit v over
# [logit ε, logit(1 − ε)] (width ≈ 72): 72/2⁶⁸ < 10⁻¹⁹.
_BISECT_STEPS: int = 68

# --- Kendall's τ: the series of (★★) (module docstring) --------------------
# Terms of the one-variable expansion in the smaller argument s: |s| ≤ 1/2 and
# the radius is ≥ 1, so each term at most halves; 64 gives ≤ 2⁻⁶⁴.
_J_MAX: int = 64
# Total degree of the double series, used when both |δ| and |κ| are < 1/2:
# its terms are O((1/2)^{m−1}), so 60 gives ≤ 2⁻⁵⁹.
_M_MAX: int = 60
# The switch: above it the series in the *other* argument is used, below it
# the double series. 1/2 also bounds the cancellation of ψ(2 + g) − ψ(2),
# the only subtraction left, at ε/g ≤ 4.5·10⁻¹⁶.
_SMALL: float = 0.5


#: ``ζ(j + 1, 2) = ζ(j + 1) − 1``, j = 0 … _J_MAX (j = 0 is +∞ and unused).
_ZETA_AT_2 = np.array([math.inf] + [float(zeta(j + 1, 2.0)) for j in range(1, _J_MAX + 1)])

#: Rows of the double series: ``_DBL[m]`` holds the coefficients of
#: ``κ^{m−i} δ^{i−1}``, i = 1 … m, i.e.
#: ``(−1)^{m+1} (ζ(m+1) − 1) C(m, i)/(m − i + 1)``.
_DBL = [np.zeros(0)] + [
    np.array([(-1.0) ** (m + 1) * float(zeta(m + 1, 2.0)) * math.comb(m, i) / (m - i + 1)
              for i in range(1, m + 1)])
    for m in range(1, _M_MAX + 1)
]


def _bb7_w_series(g: float, s: float) -> float:
    """``W = L/(δκ)`` by the one-variable expansion in ``s`` (module docstring).

    ``(g, s)`` is ``(δ, κ)`` or ``(κ, δ)`` — L is symmetric — with |g| ≥ 1/2,
    which bounds both the only subtraction left (``ψ(2 + g) − ψ(2)``, relative
    error ≤ ε/|g|) and the series ratio (|s| ≤ |g| is *not* required; what is
    required is |s| ≤ 1/2 relative to a radius ≥ 1, see the docstring).
    """
    total = float(digamma(2.0 + g) - digamma(2.0))
    power = 1.0
    for j in range(1, _J_MAX + 1):
        power *= s
        term = ((-1.0) ** (j + 1) * power
                * (float(zeta(j + 1, 2.0 + g)) - _ZETA_AT_2[j]) / (j + 1))
        total += term
        if abs(term) <= 1e-18 * max(abs(total), 1e-300):
            break
    return -total / g


def _bb7_w_double(de: float, ka: float) -> float:
    """``W = L/(δκ)`` by the double series, for |δ| and |κ| both < 1/2."""
    ka_pow = np.ones(_M_MAX)
    de_pow = np.ones(_M_MAX)
    for n in range(1, _M_MAX):
        ka_pow[n] = ka_pow[n - 1] * ka
        de_pow[n] = de_pow[n - 1] * de
    total = 0.0
    for m in range(1, _M_MAX + 1):
        row = float(np.dot(_DBL[m], ka_pow[m - 1::-1][:m] * de_pow[:m]))
        total += row
        if m > 4 and abs(row) <= 1e-18 * max(abs(total), 1e-300):
            break
    return -total


def _bb7_w(de: float, ka: float) -> float:
    """``W(δ, κ) = L/(δκ)``, analytic at δ = 0 and at κ = 0 (module docstring).

    Expand in κ when δ ≥ 1/2 (radius 2 in κ, ratio ≤ 1/2), in δ when instead
    |κ| ≥ 1/2 (radius 1 + 2/θ > 1 and δ < 1/2), else use the double series.
    Choosing on the *thresholds* and not on which of the two is larger
    matters: at θ = 194, δ = 0.98 the κ-expansion halves per term while the
    δ-expansion sits at 0.98 of its radius 1.01 and does not converge.
    """
    if de >= _SMALL:
        return _bb7_w_series(de, ka)
    if abs(ka) >= _SMALL:
        return _bb7_w_series(ka, de)
    return _bb7_w_double(de, ka)


def _bb7_tau_from_theta(theta: float, delta: float) -> float:
    """Kendall's τ of BB7 at (θ ≥ 1, δ > 0) — (★★) of the module docstring.

    Both removable singularities (θ = 2 and δ → 0) are absent: ``W`` is
    analytic at each and ``(1 − e^L)/L → −1``.
    """
    theta, delta = float(theta), float(delta)
    if not (theta >= 1.0 and delta > 0.0):
        return math.nan
    if math.isinf(theta):
        return 1.0
    if theta == 1.0:
        # τ(1, δ) = δ/(δ + 2) — Clayton's own closed form, which (★★) does
        # reproduce to rounding but not bit for bit. Used directly so that the
        # Clayton member's τ *is* Clayton's, as BB6's Joe member's θ is Joe's,
        # and so that the constructor's admissibility test δ/(δ + 2) ≤ τ and
        # this map agree exactly on the boundary.
        return delta / (delta + 2.0)
    kappa = 2.0 / theta - 1.0
    w = _bb7_w(delta, kappa)
    ell = w * delta * kappa
    ratio = -1.0 if ell == 0.0 else -math.expm1(ell) / ell
    return 1.0 - (2.0 / theta) * w * ratio


def _bb7_delta_from_tau(tau: float, theta: float) -> float:
    """Invert τ(θ, ·) at fixed θ — Brent on ``ln δ``.

    τ is strictly increasing in δ at every fixed θ (verified at 50 digits over
    θ ∈ [1, 10⁵] × δ ∈ [10⁻⁸, 10⁴]), from ``τ_Joe(θ)`` as δ → 0⁺ to 1 as
    δ → ∞, so the root is unique — which is exactly why δ, and not θ, is the
    parameter this family *recovers* (module docstring, "Parametrisation":
    τ is **not** monotone in θ).

    θ = 1 is answered by Clayton's own closed form ``δ = 2τ/(1 − τ)``, bit for
    bit, rather than by Brent. A τ the bracket cannot separate from the Joe
    limit is answered with :data:`_DELTA_FLOOR`, which moves τ by less than
    10⁻¹⁴ and is Joe to machine precision (λ_L = 2^{−10¹⁴} = 0.0).
    """
    tau, theta = float(tau), float(theta)
    if theta == 1.0:
        return 2.0 * tau / (1.0 - tau)               # the Clayton member

    def gap(x: float) -> float:
        return _bb7_tau_from_theta(theta, math.exp(x)) - tau

    lo, hi = math.log(_DELTA_FLOOR), math.log(_DELTA_CEIL)
    if gap(lo) >= 0.0:
        return _DELTA_FLOOR                          # the Joe end
    if gap(hi) <= 0.0:
        return _DELTA_CEIL
    return math.exp(brentq(gap, lo, hi, xtol=1e-15, rtol=8.9e-16, maxiter=200))


# ---------------------------------------------------------------------------
# Log-space kernel on ka = ln(1 − u), kb = ln(1 − v) (module docstring)
# ---------------------------------------------------------------------------

def _log_or_ninf(x: float) -> float:
    """``ln x`` for x > 0, ``−∞`` at x ≤ 0 — ``math.log`` raises there.

    Used for ``ln(θ − 1)``, which ``logaddexp`` then absorbs exactly at the
    Clayton boundary θ = 1.
    """
    return math.log(x) if x > 0.0 else -math.inf


def _bb7_log_x_minus_one(lp, de):
    """``ln(a^{−δ} − 1)`` from ``lp = ln(−ln a)`` (module docstring).

    ``lx = −δ ln a = δ e^{lp} ≥ 0`` and ``ln(e^{lx} − 1) = lx + ln1mexp(lx)``,
    except beyond ``lp = −_Z_ASYMPTOTIC`` where ``lx`` underflows to 0 (whose
    ``ln1mexp`` is −∞) while the true value is ``ln δ + lp`` to a relative
    10⁻³⁰⁴.
    """
    lp = np.asarray(lp, dtype=float)
    with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
        lx = de * np.exp(np.maximum(lp, -_Z_ASYMPTOTIC))
        return np.where(lp < -_Z_ASYMPTOTIC, math.log(de) + lp, lx + _log1mexp(lx))


def _bb7_terms_k(ka, kb, th, de):
    """``(lp, lq, lT, lK, lnR)`` from ``ka = ln(1 − u)``, ``kb = ln(1 − v)``.

    ``lp = ln(−ln a)``, ``lq = ln(−ln b)``, ``lT = ln T``, ``lK = ln(1 − R)``
    and ``lnR = −ln T/δ`` with ``T = a^{−δ} + b^{−δ} − 1``, ``R = T^{−1/δ}``.
    """
    lp = _bb6_logp(ka, th)
    lq = _bb6_logp(kb, th)
    ltm1 = np.logaddexp(_bb7_log_x_minus_one(lp, de), _bb7_log_x_minus_one(lq, de))
    with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
        lt = np.logaddexp(0.0, ltm1)
        # ln K: ln1mexp(ln T/δ) except where ln T itself underflows to 0.0,
        # i.e. ln(T − 1) < −700, where K = (T − 1)/δ to a relative 10⁻³⁰⁴.
        lk = np.where(ltm1 < -_Z_ASYMPTOTIC,
                      ltm1 - math.log(de),
                      _log1mexp(np.where(ltm1 < -_Z_ASYMPTOTIC, 1.0, lt / de)))
        return lp, lq, lt, lk, -lt / de


def _bb7_logpdf_k(ka, kb, th, de):
    """``ln c(u, v)`` from ``ka = ln(1 − u)``, ``kb = ln(1 − v)``."""
    lp, lq, lt, lk, lnr = _bb7_terms_k(ka, kb, th, de)
    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        # ln B, B = (θ − 1)R + θ(1 + δ)K — two non-negative terms, the first
        # −∞ at θ = 1 (Clayton), absorbed exactly rather than cancelled.
        log_b = np.logaddexp(_log_or_ninf(th - 1.0) + lnr,
                             math.log(th * (1.0 + de)) + lk)
        # (−δ − 1)(ln a + ln b) = (δ + 1)(e^{lp} + e^{lq}), never a subtraction.
        return ((th - 1.0) * (ka + kb)
                + (de + 1.0) * (np.exp(lp) + np.exp(lq))
                + (1.0 / th - 2.0) * lk
                + (-1.0 / de - 2.0) * lt
                + log_b)


def _bb7_cdf_k(ka, kb, th, de):
    """``(C, 1 − C)`` with ``1 − C = K^{1/θ} = exp(ln K/θ)``."""
    e = _bb7_terms_k(ka, kb, th, de)[3] / th
    return -np.expm1(e), np.exp(e)


def _bb7_logh_k(kb, ka, th, de):
    """``ln h(v|u)`` from ``kb = ln(1 − v)``, ``ka = ln(1 − u)``.

    ``ln h = (1/θ − 1) ln K + (θ − 1) ka + (δ + 1)(−ln a) + (−1/δ − 1) ln T``.
    """
    lp, _, lt, lk, _ = _bb7_terms_k(ka, kb, th, de)
    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        return ((1.0 / th - 1.0) * lk
                + (th - 1.0) * ka
                + (de + 1.0) * np.exp(lp)
                + (-1.0 / de - 1.0) * lt)


def _bb7_h_k(kb, ka, th, de):
    """``(h, 1 − h)`` of h(v|u), through :func:`_bb7_logh_k`."""
    logh = _bb7_logh_k(kb, ka, th, de)
    return np.exp(logh), -np.expm1(logh)


def _bb7_inv_h_k(lw, ka, th, de):
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
    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        below = lw <= _bb7_logh_k(math.log1p(-EPS), ka, th, de)
        above = lw >= _bb7_logh_k(math.log(EPS), ka, th, de)
        for _ in range(_BISECT_STEPS):
            mid = 0.5 * (lo + hi)
            kb = -np.logaddexp(0.0, mid)            # ln(1 − expit(mid))
            up = _bb7_logh_k(kb, ka, th, de) < lw
            lo = np.where(up, mid, lo)
            hi = np.where(up, hi, mid)
        x = 0.5 * (lo + hi)
        x = np.where(below, x_lo, np.where(above, -x_lo, x))
        kb = -np.logaddexp(0.0, x)
    return -np.expm1(kb), np.exp(kb)


# ---------------------------------------------------------------------------
# Copula class
# ---------------------------------------------------------------------------

class CopulaBB7(CopulaVirt):
    """BB7 (Joe-Clayton) copula — both tails, two parameters.

    Parameters
    ----------
    tau_k : float
        Kendall's τ ∈ (0, 1). Must satisfy τ > τ_Joe(θ) — the family's own
        δ → 0 limit — so that a Clayton-side exponent δ > 0 exists.
    theta7 : float, optional
        Joe-side exponent θ ≥ 1 (default 1.0, the **Clayton** member, which is
        admissible at every registered τ). θ = 1 is Clayton at δ = 2τ/(1 − τ);
        θ → θ_Joe(τ) is the Joe limit δ → 0; at fixed τ a larger θ moves
        dependence from the lower tail (λ_L = 2^{−1/δ}) to the upper one
        (λ_U = 2 − 2^{1/θ}). Named ``theta7``, not ``delta``/``delta6``, so
        that the joint fitter cannot route BB7 through BB1's or BB6's branch
        (module docstring). **θ, not δ, is the parameter this family stores**,
        because τ is monotone in δ at fixed θ but *not* in θ at fixed δ
        (module docstring, "Parametrisation").
    """

    n_params: int = 2

    def __init__(self, **kwargs):
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    # ------------------------------------------------------------------
    # Parameters — the joint (τ, θ) constraint (module docstring)
    # ------------------------------------------------------------------

    @classmethod
    def theta_max(cls, tau: float) -> float:
        """Largest θ admissible at this τ: the biggest **double** with
        ``τ_Joe(θ) < τ``, i.e. δ > 0.

        ``_joe_theta_from_tau(τ)`` is the exact boundary — the Joe limit
        δ = 0, which is *not* a BB7 — but stepping down from it by ulps does
        not find the answer: τ_Joe is flat over *many* consecutive doubles
        once θ is large (at τ = 0.99, θ_Joe = 198.71 and hundreds of ulps of θ
        share the same τ_Joe), which is why this bisects on the constructor's
        own strict test instead and only then walks up by ulps. The hook and
        the constructor then agree to the last bit. The end is attained as a
        *double* (a θ one ulp below the Joe boundary is a BB7 with a very
        small δ), so no ``TAU_PAD_REL`` pull-in is needed, unlike BB1's.
        """
        tau = float(tau)
        if not 0.0 < tau < 1.0:
            return 1.0

        def ok(t: float) -> bool:
            return t >= 1.0 and _joe_tau_from_theta(t) < tau

        hi = _joe_theta_from_tau(min(tau, 1.0 - 1e-15))
        if not hi > 1.0:
            return 1.0
        if not ok(hi):
            lo = 1.0                                 # τ_Joe(1) = 0 < τ
            for _ in range(100):                     # bisect down to one ulp
                mid = 0.5 * (lo + hi)
                if not lo < mid < hi:
                    break
                if ok(mid):
                    lo = mid
                else:
                    hi = mid
            hi = lo
        for _ in range(64):                          # up: last double that passes
            nxt = float(np.nextafter(hi, math.inf))
            if not ok(nxt):
                break
            hi = nxt
        return max(1.0, float(hi))

    @classmethod
    def _repaired_theta(cls, tau: float, theta: float) -> float:
        """``theta`` clamped into ``[1, theta_max(τ)]`` **and** checked.

        The clamp alone is not enough. ``_joe_tau_from_theta`` is a series
        evaluation, so on the plateau where τ_Joe is flat to the last bit —
        several consecutive doubles share one τ_Joe once θ is large — the
        predicate ``τ_Joe(θ) < τ`` is not monotone in θ: a θ *below*
        ``theta_max(τ)`` can be refused although ``theta_max(τ)`` itself is
        not (found at τ = 0.5, θ = 2.8562572119508050, seven ulps under
        θ_Joe). Both fallbacks are always admissible: ``theta_max(τ)`` passes
        the test by construction, and θ = 1 has τ_Joe(1) = 0 < τ.
        """
        out = min(max(float(theta), 1.0), cls.theta_max(tau))
        if cls._theta_error(tau, out) is None:
            return out
        hi = cls.theta_max(tau)
        return hi if cls._theta_error(tau, hi) is None else 1.0

    @classmethod
    def constrain_params(cls, params: dict) -> dict:
        """Clamp θ into ``[1, theta_max(τ)]`` — a box cannot express it.

        The projection a bounded optimiser needs (BB1's and BB6's precedent);
        the joint MLE does not rely on it, because ``_two_parameter_spec``'s
        ``theta7`` branch optimises in (θ, δ), whose box *is* the admissible
        set.
        """
        out = dict(params)
        tau = float(out.get("tau_k", 0.0))
        theta = float(out.get("theta7", _THETA_DEFAULT))
        if not (np.isfinite(tau) and 0.0 < tau < 1.0):
            out["theta7"] = max(1.0, theta) if np.isfinite(theta) else _THETA_DEFAULT
            return out
        out["theta7"] = cls._repaired_theta(tau, theta if np.isfinite(theta) else 1.0)
        return out

    @classmethod
    def constructible_params(cls, params: dict) -> dict:
        """``params`` itself when BB7 builds at it; else θ moved into
        ``[1, theta_max(τ)]``.

        The test is the constructor's own (:meth:`_update_params`, the same
        floating-point operations). Only a refused pair changes, and only θ —
        τ is the value the caller drew (multistart jitter, family draw) and
        stays. Both ends are admissible doubles here, so the repaired θ goes
        **on** the end without BB1's ``TAU_PAD_REL`` pull-in. A pair no θ
        repairs (τ ≥ 1, τ ≤ 0, non-finite) is returned unchanged, for the
        constructor to refuse.
        """
        tau = params.get("tau_k")
        theta = params.get("theta7", _THETA_DEFAULT)
        try:
            tau_f, theta_f = float(tau), float(theta)
        except (TypeError, ValueError):
            return params
        if isinstance(theta, (bool, np.bool_)):     # the constructor refuses it
            return params
        if cls._theta_error(tau_f, theta_f) is None:
            return params
        if not (np.isfinite(tau_f) and 0.0 < tau_f < 1.0):
            return params
        return {**params, "theta7": cls._repaired_theta(tau_f, theta_f)}

    @classmethod
    def _theta_error(cls, tau: float, theta: float) -> str | None:
        """Why the constructor refuses (τ, θ), or ``None`` when it accepts."""
        if not np.isfinite(theta) or theta < 1.0:
            return f"BB7: theta7 must be >= 1, got {theta!r}."
        if not 0.0 < float(tau) < 1.0:
            return f"BB7: tau_k={tau!r} must be in (0, 1)."
        if not _joe_tau_from_theta(theta) < float(tau):
            return (f"BB7: tau_k={tau!r} is not reachable with theta7={theta!r} — "
                    f"BB7 needs tau > tau_Joe(theta7) = "
                    f"{_joe_tau_from_theta(theta):.6g} (theta7 < theta_Joe(tau) = "
                    f"{_joe_theta_from_tau(min(float(tau), 1.0 - 1e-15)):.6g}), so "
                    f"that the Clayton-side exponent delta > 0 exists.")
        return None

    def _update_params(self):
        tau = float(self.params['tau_k'])
        theta = self.params.get('theta7', _THETA_DEFAULT)
        if isinstance(theta, (bool, np.bool_)):
            raise CopulaParameterError(f'BB7: theta7={theta!r} is not a number.')
        try:
            theta = float(theta)
        except (TypeError, ValueError):
            raise CopulaParameterError(f'BB7: theta7={theta!r} is not a number.') from None
        err = self._theta_error(tau, theta)
        if err is not None:
            raise CopulaParameterError(err)
        self.theta = theta
        self.delta7 = _bb7_delta_from_tau(tau, theta)
        self.theta7 = theta
        self.params['theta7'] = theta

    def tau_of(self) -> float:
        """τ recomputed from (θ, δ) by (★★) — the forward map of the constructor."""
        return _bb7_tau_from_theta(self.theta, self.delta7)

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
        return _bb7_logpdf_k(ka, kb, self.theta, self.delta7)

    def _k_cdf(self, ka, kb):
        return _bb7_cdf_k(ka, kb, self.theta, self.delta7)

    def _k_h(self, kb, ka):
        return _bb7_h_k(kb, ka, self.theta, self.delta7)

    def _k_inv_h(self, lw, ka):
        return _bb7_inv_h_k(lw, ka, self.theta, self.delta7)

    # ------------------------------------------------------------------
    # CDF / PDF / h-function
    # ------------------------------------------------------------------

    def cdf(self, uv):
        """C(u,v) = 1 − K^{1/θ}, K = 1 − T^{−1/δ} (module docstring)."""
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
        """c(u,v) = (ū v̄)^{θ−1}(ab)^{−δ−1} K^{1/θ−2} T^{−1/δ−2} B.

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
        """h(v|u) = K^{1/θ−1} ū^{θ−1} a^{−δ−1} T^{−1/δ−1}."""
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
        """λ_L = 2^{−1/δ}, λ_U = 2 − 2^{1/θ} (module docstring)."""
        return (float(2.0 ** (-1.0 / self.delta7)),
                float(2.0 - 2.0 ** (1.0 / self.theta)))

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    @classmethod
    def fit(cls, data: np.ndarray, method: str = 'mle', *, weights=None,
            pseudo_obs: bool = False):
        """Joint MLE of (τ, θ) by default; ``method='tau'`` is itau.

        Kendall's τ alone cannot identify θ: ``'tau'`` inverts τ̂ and fits
        θ by maximum likelihood at that τ (a profile likelihood, FR-12 —
        before, it logged a warning and ran the joint MLE). ``weights`` and
        ``pseudo_obs`` as in :meth:`CopulaVirt.fit`.
        """
        return super().fit(data, method=method, weights=weights, pseudo_obs=pseudo_obs)


if __name__ == '__main__':
    from pathlib import Path

    from pmcprg.copulas.archimedean.clayton import CopulaClayton
    from pmcprg.copulas.archimedean.joe import CopulaJoe

    cop = CopulaBB7(tau_k=0.6, theta7=1.5)
    lL, lU = cop.tail_dependence()
    print(f'Copula : {cop.copula_enum.value.LONG_NAME}')
    print(f'tau_k  : {cop.params["tau_k"]:.4f}  theta7={cop.theta:.4f}  '
          f'delta(recovered)={cop.delta7:.6f}')
    print(f'tau    check (★★): {cop.tau_of():.15f}  (want {cop.params["tau_k"]:.15f})')
    print(f'pdf(0.3, 0.7) = {cop.pdf([0.3, 0.7]):.6f}')
    print(f'cdf(0.3, 0.7) = {cop.cdf([0.3, 0.7]):.6f}')
    print(f'h(0.7 | 0.3)  = {cop.conditional_cdf(0.7, 0.3):.6f}')
    print(f'tail dep : λ_L = {lL:.4f} [= 2^(-1/δ)],  λ_U = {lU:.4f} [= 2 − 2^(1/θ)]')

    print('\nθ = 1 vs Clayton @ (0.3, 0.7), τ = 0.5:')
    print(f'  BB7.pdf={CopulaBB7(tau_k=0.5, theta7=1.0).pdf([0.3, 0.7]):.12f}  '
          f'Clayton.pdf={CopulaClayton(tau_k=0.5).pdf([0.3, 0.7]):.12f}')
    theta_joe = CopulaBB7.theta_max(0.6)
    print('θ → θ_Joe(τ) (δ → 0) vs Joe @ (0.3, 0.7), τ = 0.6:')
    print(f'  BB7.pdf={CopulaBB7(tau_k=0.6, theta7=theta_joe).pdf([0.3, 0.7]):.12f}  '
          f'Joe.pdf={CopulaJoe(tau_k=0.6).pdf([0.3, 0.7]):.12f}')

    plot_dir = Path('./data/Plots/Copulas')
    plot_dir.mkdir(parents=True, exist_ok=True)
    cop.plot_pdf(plot_dir)
    cop.plot_cdf(plot_dir)
    cop.plot_h_function(plot_dir)
    cop.plot_samples(plot_dir)
    cop.plot_overview(plot_dir)
