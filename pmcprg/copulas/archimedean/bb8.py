"""
BB8 copula (Joe-Frank) — no tail dependence for δ < 1, two parameters (FR-9).

Generator and CDF (re-derived, not transcribed)
-----------------------------------------------
The generator VineCopula and vinecopulib use for family 10 (BB8) is

    φ(t) = − ln( [1 − (1 − δ t)^θ] / [1 − (1 − δ)^θ] ),    θ ≥ 1,  0 < δ ≤ 1,

and the first thing checked here was whether it *is* an Archimedean generator
on [0, 1] rather than whether it matches a printed formula. Write
``E = 1 − δ`` and ``η = 1 − E^θ ∈ (0, 1]``. Then

* **φ(1) = 0** exactly, and not as a limit: at t = 1 the numerator is
  ``1 − (1 − δ)^θ = η``, the very quantity the denominator normalises by, so
  the ratio is 1. The δ inside ``(1 − δ t)`` and the ``(1 − δ)^θ`` of the
  denominator are the *same* boundary value, which is what makes the
  normalisation work at all.
* **φ(0) = +∞**: at t = 0 the numerator is ``1 − 1 = 0``. The generator is
  therefore **strict** for every admissible (θ, δ) — including δ < 1, where
  one might expect the ``(1 − δ t)`` bracket to cut the domain short. It does
  not: ``1 − δt`` stays in ``[E, 1]``, away from 0, and it is the *numerator*
  that vanishes at t = 0. (This was the audit's suspected trouble spot; it is
  not one. What does resist closed form is the τ integral — below — and for a
  reason that has nothing to do with a boundary of the domain.)
* φ is continuous, strictly decreasing and convex on (0, 1] and
  ``C = φ^{-1}(φ(u) + φ(v))`` has uniform margins and a strictly positive
  density on the open square (checked at 50 digits in ``mpmath`` on
  θ ∈ {1, 1.3, 2, 5, 12} × δ ∈ {0.05, 0.3, 0.7, 0.999, 1}: margins to
  6.5·10⁻¹⁷, ``min ∂²C/∂u∂v`` 0.0498 at the worst pair tried).

Inverting — ``e^{−s} = (1 − (1 − δt)^θ)/η`` ⇒ ``(1 − δt)^θ = 1 − η e^{−s}`` —

    φ^{-1}(s) = [ 1 − (1 − η e^{−s})^{1/θ} ] / δ ,

so with ``P = 1 − δu``, ``Q = 1 − δv`` and ``A = (1 − P^θ)(1 − Q^θ)/η``,

    C(u, v) = [ 1 − (1 − A)^{1/θ} ] / δ .

Checked against ``φ^{-1}(φ(u) + φ(v))`` evaluated independently at 60 digits:
relative agreement ≤ 9.7·10⁻⁵² over the same grid.

Sub-models — and the nickname, which is only half right
--------------------------------------------------------
* **δ = 1 → Joe**, *exactly* and with no limit to take: ``(1 − δ)^θ = 0``, so
  the denominator is 1 and ``φ(t) = −ln(1 − (1 − t)^θ)`` is Joe's own
  generator (``pmcprg.copulas.archimedean.joe``). Verified to 59 digits
  (``max |C_BB8 − C_Joe| ≤ 1.4·10⁻⁵⁹`` at θ ∈ {1.5, 3, 8}). δ = 1 is the
  **upper end** of the admissible δ ∈ (0, 1]: a *boundary* sub-model, hence
  the one-sided null in ``_stderr._SUBMODEL_NESTING``.
* **θ = 1 → the independence copula Π, not Frank.** The popular name
  "Joe–Frank" invites the guess that θ = 1 leaves Frank's family, and it is
  wrong: at θ = 1 the numerator is ``1 − (1 − δt) = δt`` and the denominator
  ``δ``, so ``φ(t) = −ln t`` whatever δ is. Verified at 60 digits
  (``|C − uv| ≤ 1.7·10⁻¹⁶`` at δ ∈ {0.1, 0.5, 1}) and in double on the whole
  corner grid (``ln c`` identically 0 to 10⁻¹⁵, C to uv to 10⁻¹⁵, at
  δ ∈ {0.02, 0.2, 0.5, 0.9, 1}). θ = 1 is the **lower end** of θ ≥ 1, so it too is a
  boundary — but of the *one-parameter-free* kind ``independence_lr_test``
  already covers, not a two-parameter nesting, and it is therefore not
  registered in ``_SUBMODEL_NESTING`` (the Product copula has no free τ to
  profile against).
* **δ → 0 at fixed θ → independence** as well, linearly in δ: on the whole
  corner grid ``max |C/uv − 1| = 1.5·10⁻³, 1.5·10⁻⁶, 1.5·10⁻⁹`` at θ = 4 and
  δ = 10⁻³, 10⁻⁶, 10⁻⁹ (5.5·10⁻³, 5.5·10⁻⁶, 5.5·10⁻⁹ at θ = 12). So the
  family degenerates at *both* of θ = 1 and δ → 0, and the two degeneracies
  are the same copula.
* **Frank is a limit, not a member**: it appears only on the joint path
  ``θ → ∞, δ → 0 with θδ = κ`` held fixed, where
  ``(1 − δt)^θ → e^{−κt}`` and φ becomes
  ``−ln[(1 − e^{−κt})/(1 − e^{−κ})]`` — Frank's generator at parameter κ > 0
  (``pmcprg.copulas.archimedean.frank``; Frank's usual
  ``−ln[(e^{−κt} − 1)/(e^{−κ} − 1)]`` is the same expression, both fractions
  being ratios of the *same* two negatives). Verified numerically:
  ``max |C_BB8 − C_Frank|`` falls as 1/θ — 1.2·10⁻², 9.9·10⁻⁴, 9.7·10⁻⁶,
  9.7·10⁻⁸ at θ = 10, 10², 10⁴, 10⁶ for κ = 5. Only the *positive-dependence*
  half of Frank is reached (κ > 0), and only in the limit: no finite (θ, δ)
  in the admissible set is a Frank copula. "Joe–Frank" is therefore accurate
  as "interpolates between Joe and (the positive half of) Frank", and
  inaccurate as a statement about any single parameter value.

Kendall's τ — a hypergeometric closed form, and why it is still a quadrature
----------------------------------------------------------------------------
The audit's prior note says BB8 "has no closed-form τ". That is right about
elementary functions and wrong about special ones; both halves were checked
rather than assumed.

Genest & MacKay (1986) give ``τ = 1 + 4∫₀¹ φ/φ′``. With ``x = 1 − δt``, then
``y = x^θ``, then ``w = 1 − y = ηr``, the integral collapses to

    τ(θ, δ) = 1 + (4 η² / (θ²δ²)) ∫₀¹ r ln r (1 − η r)^{2/θ − 2} dr .      (†)

``sympy`` returns (†) unevaluated — indefinite and definite, with and without
``meijerg`` — and the one piece it does evaluate comes back as a Lerch
transcendent, which is the shape of the answer: expanding ``(1 − ηr)^a`` and
using ``∫₀¹ r^{k+1} ln r dr = −1/(k+2)²`` gives

    τ(θ, δ) = 1 − ( η² / (θ²δ²) ) · ₃F₂( 2, 2, 2 − 2/θ ; 3, 3 ; η ) .      (‡)

(‡) is exact — verified against a 50-digit ``mpmath`` quadrature of φ/φ′
itself at θ ∈ {1.5, 2, 4, 20} × δ ∈ {0.1, 0.5, 0.9}: agreement 6·10⁻⁴⁴ to
9·10⁻²⁶. It is not, however, a usable closed form. The series converges only
for ``Σb − Σa = 2/θ > 0``, i.e. like ``Σ k^{−1−2/θ}``, so it needs of the
order of ``10^{16θ/2}`` terms for double precision at large θ; ``mpmath``'s
own ``hyper`` raises ``NoConvergence`` at θ ≥ 1000 with the default term
budget, and there is no ₃F₂ in ``scipy``. **τ is therefore evaluated by
quadrature here**, which confirms the audit's note in the only sense that
matters to the code, while correcting it in principle.

*The quadrature.* The substitution that makes (†) cheap is not r but
``z = −θ ln x`` followed by ``m = 2z/θ``, which puts the integrand on the
scale the family actually varies on:

    τ = 1 + (2/(θδ²)) ∫₀^M G(m) dm,   G(m) = A₁ ln(A₁/η) e^{m(θ/2 − 1)},
    A₁ = 1 − e^{−θm/2},   M = −2 ln(1 − δ)  (independent of θ),
    η = 1 − e^{−θM/2}.

``G ≤ 0``, ``G(0) = G(M) = 0``, and ``G(m) → −e^{−m}`` in the bulk whatever θ
is — the two large factors ``e^{m(θ/2−1)}`` and ``ln(A₁/η) ≈ −e^{−θm/2}``
cancel exactly. The only structure left is a boundary layer of width ``2/θ``
at each end, which the package's own :func:`~pmcprg.copulas.extreme_value.
_pickands.graded_mesh` (decades ``10^{-k}`` and ``1 − 10^{-k}``) resolves.
:func:`_bb8_tau_quad` is a 24-node composite Gauss–Legendre rule on that mesh
after ``m = M s``: **36 panels, 864 vectorised integrand evaluations, 0.077 ms
per τ** (measured, Apple M-series, one NumPy call).

*Accuracy, measured not estimated.* Against a 60-digit ``mpmath`` reference
built on the same identity but with two different geometric breakpoint sets
(ratios 10 and 3), themselves agreeing to 10⁻⁵², on the grid
θ ∈ {1+10⁻⁷, 1.001, 1.1, 1.5, 2, 3, 7, 25, 10², 10³, 10⁴, 10⁵, 10⁶, 10⁷} ×
δ ∈ {10⁻³, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.999, 1−10⁻⁶, 1−10⁻¹²}:
**max absolute error 6.0·10⁻¹⁶** (at θ = 100, δ = 0.01), max *relative* error
3.9·10⁻⁵ — the latter only where τ itself is 1.1·10⁻¹¹ (θ = 1 + 10⁻⁷,
δ = 10⁻³), i.e. still 4·10⁻¹⁶ in absolute terms. τ is thus accurate to the
last bit of a double everywhere the family is fitted.

A warning worth recording, because it cost this round two wrong reference
values: ``ln(A₁/η)`` computed as ``ln A₁ − ln η`` cancels catastrophically
(at θ = 1000, δ = 0.2 it loses 44 of 50 digits), and *mpmath is not immune* —
a naive 50-digit reference disagreed with the truth in the third digit while
looking perfectly converged. Both the double-precision kernel and the
``mpmath`` reference in ``test_bb8.py`` use the cancellation-free
``ln(A₁/η) = log1p(−(e^{−b} − e^{−B})/η)`` with the ``y ≥ ½`` branch falling
back to ``ln η − ln A₁`` (where that difference is ≥ ln 2 and so safe).

Parametrisation — and the joint constraint BB1, BB6 and Tawn have but BB8 does not
----------------------------------------------------------------------------------
``tau_k`` is stored and θ recovered from (τ, δ), as for BB1/BB6/Tawn. Unlike
all three, **(τ, δ) is not jointly constrained**: at every fixed δ ∈ (0, 1],
τ(θ, δ) rises continuously from 0 at θ = 1 to 1 as θ → ∞, so the admissible
set is the full rectangle ``θ ≥ 1, 0 < δ ≤ 1`` and the whole registered
τ-range ``(0, 1)`` is reachable at every δ. This was verified, not assumed:
τ(10⁷, δ) = 0.99960, 0.99996, 0.999992, 0.999998, 0.9999994, 0.9999998 at
δ = 10⁻³, 0.01, 0.05, 0.2, 0.5, 1. :meth:`CopulaBB8.constructible_params`
therefore only has to clip δ into (0, 1]; there is no τ-dependent repair of
the BB1/BB6/Tawn kind, and :meth:`constrain_params` is the plain box.

What *does* bite is how slowly τ approaches 1 once δ is small. The rate is
Joe's — ``1 − τ = O(1/θ)`` — but the constant blows up: fitting the measured
``(1 − τ)·θ`` at θ = 10⁷ gives

    1 − τ  ≃  (4/δ − 2) / θ        (θ → ∞),

which reproduces 2.000000, 2.444444, 6.000000, 18.00000, 77.9998, 397.993 at
δ = 1, 0.9, 0.5, 0.2, 0.05, 0.01 — relative agreement 1.3·10⁻⁷ to 1.6·10⁻⁵,
the residual being the ``1/θ²`` term, and δ = 1 giving Joe's own ``1 − 2/θ``.
(This corrects a ``θ^{-2}`` rate guessed earlier in this round from a
mis-converged reference; the rate is ``θ^{-1}``, and it is the *constant* that
carries δ.) θ is therefore capped at :data:`_THETA_HI` = 10⁷ and a τ beyond
``τ(10⁷, δ)`` builds at the cap and **stores the τ it realises** — Galambos's
and Tawn's RB-10 convention, not a silent clip. :meth:`reachable_tau_bounds`
reports the largest such cap, the δ = 1 (Joe) one, exactly as Tawn reports its
ψ = 1 one.

*Why the name ``delta8`` and not ``delta``.* ``_fit._two_parameter_spec``
dispatches on the **parameter name**: ``"delta"`` is BB1's branch and carries
BB1's ``τ = 1 − 2/(δ(θ+2))`` map, ``"delta6"`` is BB6's. A BB8 routed through
either would be fitted with the wrong parametrisation and never complain,
every (θ, δ) pair being numerically plausible. Same reason t-EV's ν is
``"nu"`` and not Student's ``"df"``. ``test_bb8.py`` proves BB8 takes neither
BB1's nor BB6's path.

Tail dependence
---------------
From ``1 − φ^{-1}(s) = 1 − [1 − (1 − η e^{−s})^{1/θ}]/δ``, which at s = 0
equals 0 with a **non-zero** derivative whenever δ < 1 (``η < 1`` keeps
``1 − η e^{−s}`` away from 0), the upper-tail coefficient
``λ_U = 2 − lim_{s→0} (1 − φ^{-1}(2s))/(1 − φ^{-1}(s)) = 2 − 2 = 0``. At
δ = 1 the derivative is not finite — ``1 − φ^{-1}(s) = (1 − e^{−s})^{1/θ}
≈ s^{1/θ}`` — and Joe's ``λ_U = 2 − 2^{1/θ}`` returns. So

    λ_U = 0 for δ < 1,   λ_U = 2 − 2^{1/θ} at δ = 1,    λ_L = 0 always,

a genuine **discontinuity at δ = 1**, not a modelling choice: the family has
no upper-tail dependence at all except on that one edge. λ_L = 0 because
``φ^{-1}(s) ≈ η e^{−s}/(δθ)`` as s → ∞, so ``φ^{-1}(2s)/φ^{-1}(s) → 0``.
Both checked numerically on the diagonal: ``2 − (1 − C(u,u))/(1 − u)`` at
``1 − u = 10^{-k}``, k = 3…8, lands on λ_U and, for δ < 1, falls a decade per
decade — the λ_U = 0 signature; ``C(u,u)/u`` at ``u = 10^{-k}``, k = 6…12,
stays under 10⁻⁴ for λ_L. k stops at 8 for the upper tail because below that
the double's resolution of ``ln(1 − A)`` (whose scale is θ ln E) is coarser
than the quantity being measured, and the ratio walks away again — 1.7 at
1 − u = 10⁻¹⁵, τ = 0.95, δ = 0.01 (``test_bb8.py`` records this).

Densities (derivation)
----------------------
With ``P = 1 − δu``, ``Q = 1 − δv``, ``A = (1 − P^θ)(1 − Q^θ)/η``:

    h(v|u) = ∂C/∂u = P^{θ−1} · (1 − Q^θ)/η · (1 − A)^{1/θ − 1},

    c(u, v) = θ δ (P Q)^{θ−1} (1 − A)^{1/θ − 2} (1 − A/θ) / η .

Both derived by hand from C above (``∂A/∂u = θδ P^{θ−1}(1 − Q^θ)/η``) and
checked against ``mpmath`` finite differences of C. ``h(1|u) = 1`` exactly
(``Q = E`` ⇒ ``A = 1 − P^θ`` ⇒ the two powers of P cancel), and ``1 − A/θ > 0``
on the whole square **because** θ ≥ 1 and A ≤ 1 — the density is positive
exactly on the admissible set. At θ = 1 the two brackets cancel and
``c ≡ 1`` identically, the independence check above in closed form.

Numerics
--------
Everything is computed from ``ka = ln(1 − u)``, ``kb = ln(1 − v)`` in log
space; nothing is formed in linear scale. The four places this family cancels
and how each is handled:

* ``ln P``. ``P = E + δ(1 − u)``: the form ``log1p(δ·expm1(ka))`` is exact as
  u → 0 (where ``ln P ≈ −δu`` and ``ln(1 − u)`` is the *wrong* small number),
  and the form ``ka + ln(δ + E e^{−ka})`` is exact where P is small (δ → 1 and
  u → 1, where the first loses everything). The switch is at ``P = ½``, and
  at δ = 1 the second branch returns ``ka`` bit for bit, so the Joe member
  evaluates on Joe's own coordinate.
* ``ln(1 − P^θ)`` is ``log(−expm1(θ ln P))`` — exact as u → 0, where
  ``1 − P^θ ≈ θδu`` cancels to nothing if formed as ``1 − exp(...)``.
* ``P^θ − E^θ`` (inside ``1 − A``) cancels as u → 1. It is never a
  difference: ``ρ = ln(P/E) = log1p(δ(1 − u)/E)`` is cancellation-free, and
  ``ln(P^θ − E^θ) = θ ln P + ln(−expm1(−θρ))``.
* ``A`` and ``1 − A`` cancel at **opposite** corners, and each has its own
  branch (switched at A = ½, where both are safe). Towards (1, 1),
  ``ln(1 − A) = ln D − ln η`` with ``D = Q^θ(1 − P^θ) + (P^θ − E^θ)`` — a
  ``logaddexp`` of two terms that are **separately non-negative** for every
  admissible point, so the sum has no cancellation at all
  (``D = P^θ + Q^θ − (PQ)^θ − E^θ``, regrouped). Towards (0, 0) that same
  expression is a disaster: at (10⁻¹², 10⁻¹²) the true ``ln(1 − A)`` is
  −10⁻²⁴ while ``ln D`` and ``ln η`` are separately rounded numbers of order
  1, so their difference is **exactly 0** and C comes out 0 instead of
  10⁻²⁴. There ``ln A = ln(1 − P^θ) + ln(1 − Q^θ) − ln η`` is exact instead,
  and ``ln(1 − A) = log1p(−e^{ln A})`` — with the series ``−y − y²/2`` below
  ln A = −36, where ``log1p`` itself underflows to 0. (Found by the corner
  grid of ``test_bb8.py``, not suspected in advance.)
  ``1 − A/θ = (1 − A) + A(1 − 1/θ)`` is a further ``logaddexp`` of two
  non-negative terms, with ``ln(1 − 1/θ) = −∞`` at θ = 1 absorbed exactly.
* ``1 − C = ((1 − A)^{1/θ} − E)/δ`` cancels completely at (1, 1), where
  ``1 − A = E^θ`` exactly. It is computed as
  ``(E/δ)·expm1(ln(1 − A)/θ − ln E)``, whose argument is ≥ 0 everywhere and
  exactly 0 at the corner; at δ = 1 it degenerates and Joe's
  ``(1 − A)^{1/θ}`` is used directly.

Measured against an adaptive-precision ``mpmath`` ground truth (480 → 7680
digits, C from the generator alone and c, h as its finite differences at a
step 10⁻²⁰ of the distance to the edge), on the 6×6 corner grid
``u, v ∈ [10⁻¹², 1 − 10⁻¹²]`` at fourteen (τ, δ) from (10⁻⁶, 0.05) to
(0.99, 1), the Joe edge included — 504 points:

    **C ≤ 6.2·10⁻¹⁵, h ≤ 4.5·10⁻¹³, ln c ≤ 1.1·10⁻¹⁴, all relative.**

The single exclusion is ``h`` where its true value is below 10⁻²⁹⁰ and no
double can hold it (1.0·10⁻¹¹⁹⁶ at τ = 0.99, δ = 1, (u, v) = (1 − 10⁻⁶,
10⁻¹²)); ``ln c`` at that same point is right to 6·10⁻¹⁶. There is no
fallback — a value that cannot be computed is NaN, not ``EPS`` (audit RB-9).

*The reference needed hardening too, twice.* A ground truth that raises the
precision until **two** successive values agree can stop on two values that
are both wrong: at (10⁻¹², 1 − 10⁻¹²), τ = 0.99, δ = 0.5, the mixed finite
difference of C needs ≈ 230 digits of cancellation, and a 240-digit reference
returned ln c = −407.8454 where the truth is −407.836452727273792 — the
double-precision kernel was right to 2·10⁻¹⁶ and the *reference* was wrong in
the fourth digit. The measurement above therefore starts at 480 digits and
requires **three** successive precisions to agree to 10⁻²⁵.

*Inverse h-function.* No closed form: ``ln h(v|u) = ln w`` is solved by 68
steps of vectorised bisection on ``logit v``, which never forms ``1 − v`` in
linear scale (``ln(1 − v) = −logaddexp(0, logit v)``), with
:meth:`CopulaVirt.inv_h`'s saturation convention at the ends — BB6's and
Tawn's arrangement, for BB6's reason (per-point Brent takes minutes to sample
20 000 pairs).

Fitting
-------
``fit`` is always the joint two-parameter MLE; τ alone cannot identify δ, so
``method='tau'`` warns and falls back (BB1's, BB6's and Tawn's precedent).
``_two_parameter_spec``'s ``delta8`` branch optimises in ``(ln(θ − 1), δ)``,
whose box **is** the admissible set, and maps back through the memoised
``_tau_of`` so the constructor gets the very θ the optimiser proposed instead
of re-inverting τ by Brent (Tawn's memo, reused here). One ``fit()`` on
n = 1000 costs of the order of 10³ τ evaluations, i.e. ≈ 10⁶ integrand
evaluations — see the CHANGELOG for the measured figure.

References
----------
* Joe, H. (1993). Parametric families of multivariate distributions with given
  margins. *Journal of Multivariate Analysis* 46(2), 262–282,
  doi:10.1006/jmva.1993.1061 (DOI, journal, volume, issue and pages verified
  against Crossref for this round — the construction of the two-parameter
  Archimedean families, BB8 among them).
* Joe, H. (1997). *Multivariate Models and Multivariate Dependence Concepts*,
  Chapman & Hall, doi:10.1201/b13150, ch. 5 (family BB8; note that Crossref's
  registered title carries "Multivariate" twice, unlike the spine).
* Genest, C. & MacKay, J. (1986). The joy of copulas: bivariate distributions
  with uniform marginals. *The American Statistician* 40(4), 280–283,
  doi:10.1080/00031305.1986.10475414 (τ = 1 + 4∫φ/φ′, the source of (†)).
* Hofert, M., Mächler, M. & McNeil, A. J. (2012). Likelihood inference for
  Archimedean copulas in high dimensions under known margins.
  *J. Multivariate Anal.* 110, 133–150, doi:10.1016/j.jmva.2012.02.019
  (log-space evaluation of Archimedean densities).
* Mächler, M. (2012). *Accurately computing log(1 − exp(−|a|))*, vignette of
  the R package Rmpfr (the ln 2 switch, reused from ``joe.py``).
* Schepsmeier, U. et al. (2018). *VineCopula*, R package (family 10, BB8,
  same (θ, δ) convention; its own τ is numerical too).
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

from pmcprg.copulas._base import CopulaVirt
from pmcprg.copulas.archimedean.joe import (
    _joe_tau_from_theta,
    _joe_theta_from_tau,
    _log1mexp,
)
from pmcprg.copulas.extreme_value._pickands import graded_mesh
from pmcprg.exceptions import CopulaParameterError
from pmcprg.numerics import EPS, ONE_MINUS_EPS, minmaxEPS

logger = logging.getLogger(__name__)

# A block without ``delta8`` is the Joe member δ = 1, admissible at every
# registered τ (module docstring, "Parametrisation") — BB6's and Tawn's
# convention, not BB1's interior default.
_DELTA_DEFAULT: float = 1.0

# θ bracket of the τ inversion. The low end is the next double above 1 (τ
# below 10⁻¹⁵ there, under EPS for every δ); the high end is where the τ
# quadrature was measured (module docstring) and is the θ a τ beyond the
# family's reach is built at.
_THETA_LO: float = float(np.nextafter(1.0, 2.0))
_THETA_HI: float = 1.0e7

# τ quadrature (module docstring): a composite Gauss–Legendre rule on
# ``graded_mesh``, 36 panels × 24 nodes = 864 integrand evaluations.
_TAU_NODES: int = 24
_TAU_DECADES: int = 18

# ``G(m) → −e^{−m}`` in the bulk, so everything beyond m = 45 is below
# e^{−45} = 3·10⁻²⁰ of the integral and the upper end M = −2 ln(1 − δ) may be
# truncated there. M exceeds it only for δ > 1 − 1.7·10⁻¹⁰, where the bulk
# asymptote is the whole integrand.
_M_CAP: float = 45.0

# Bisection steps of the inverse h-function, on logit v over
# [logit ε, logit(1 − ε)] (width ≈ 72): 72/2⁶⁸ < 10⁻¹⁹ (BB6's rule).
_BISECT_STEPS: int = 68

_GL_NODES: dict[int, tuple[np.ndarray, np.ndarray]] = {}


# ---------------------------------------------------------------------------
# Kendall's τ — quadrature of (†) in the m coordinate (module docstring)
# ---------------------------------------------------------------------------

def _bb8_tau_quad(theta: float, delta: float) -> float:
    """τ(θ, δ) by the graded Gauss–Legendre rule of the module docstring.

    ``τ = 1 + (2/(θδ²)) ∫₀^M G(m) dm`` with
    ``G = A₁ ln(A₁/η) e^{m(θ/2 − 1)}``, ``A₁ = 1 − e^{−b}``, ``b = θm/2``,
    ``M = −2 ln(1 − δ)``, ``B = θM/2``, ``η = 1 − e^{−B}``.

    ``δ = 1`` is Joe's closed form, not a limit of this rule: B = ∞ there and
    the family *is* Joe (module docstring), so ``_joe_tau_from_theta`` gives
    the τ ``CopulaJoe`` itself would report, bit for bit.
    """
    theta = float(theta)
    delta = float(delta)
    if not theta > 1.0:
        return 0.0
    if delta >= 1.0:
        return _joe_tau_from_theta(theta)
    big_m = -2.0 * math.log1p(-delta)
    m_hi = min(big_m, _M_CAP)
    big_b = 0.5 * theta * big_m            # = −θ ln(1 − δ)
    nodes = _GL_NODES.get(_TAU_NODES)
    if nodes is None:
        nodes = _GL_NODES[_TAU_NODES] = np.polynomial.legendre.leggauss(_TAU_NODES)
    x, w = nodes
    bpts = graded_mesh(0.5, 0.5, n_decades=_TAU_DECADES)
    lo, hi = bpts[:-1, None], bpts[1:, None]
    s = (0.5 * (hi - lo) * x + 0.5 * (hi + lo)).ravel()
    wt = (0.5 * (hi - lo) * w).ravel() * m_hi
    m = s * m_hi
    b = 0.5 * theta * m
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        log_a1 = np.log(-np.expm1(-b))                      # ln(1 − e^{−b})
        log_eta = math.log(-math.expm1(-big_b)) if big_b < 700.0 else 0.0
        # ln y, y = (e^{−b} − e^{−B})/η ∈ [0, 1] — no cancellation anywhere:
        # e^{−b} − e^{−B} = e^{−b}(1 − e^{−(B−b)}).
        ly = np.minimum(
            -b + np.log(-np.expm1(np.minimum(b - big_b, -0.0))) - log_eta, 0.0)
        # −ln(A₁/η) = −ln(1 − y), by the branch the module docstring gives:
        # the series for y → 0 (where ln(1 − y) underflows), ``log1mexp`` in
        # the middle, and ln η − ln A₁ ≥ ln 2 for y ≥ ½ (where the difference
        # of logs is the *safe* form and ``log1p(−y)`` the unstable one).
        tiny = ly < -36.0
        mid = (~tiny) & (ly < -math.log(2.0))
        log_neg_l = np.where(
            tiny, ly + 0.5 * np.exp(ly),
            np.where(mid,
                     np.log(-_log1mexp(-np.where(mid, ly, -1.0))),
                     np.log(np.maximum(log_eta - log_a1, 0.0))))
        g = -np.exp(log_a1 + log_neg_l + m * (0.5 * theta - 1.0))
    integral = float(np.dot(wt, np.nan_to_num(g, nan=0.0)))
    return 1.0 + 2.0 * integral / (theta * delta * delta)


# Memo of the forward map and of its exact inverse on the values it produced —
# Tawn's arrangement, for Tawn's reason: the joint MLE optimises in θ and
# hands the constructor τ(θ), which would otherwise re-invert τ by Brent
# (≈ 40 quadratures, 3.5·10⁴ integrand evaluations) at every likelihood
# evaluation. A hit returns the θ that produced τ — a better inverse than a
# Brent root. Never used at δ = 1, which is Joe's closed form both ways.
_MEMO_SIZE = 1 << 14
_TAU_MEMO: "OrderedDict[tuple, float]" = OrderedDict()
_THETA_MEMO: "OrderedDict[tuple, float]" = OrderedDict()


def _remember(memo, key, value):
    memo[key] = value
    memo.move_to_end(key)
    if len(memo) > _MEMO_SIZE:
        memo.popitem(last=False)


def _tau_of(theta: float, delta: float) -> float:
    """τ(θ, δ), memoised; records θ as the inverse of the returned τ on *every*
    call (hit or miss), so a constructor called right after with this τ gets
    this very θ back whatever the memo held before."""
    key = (float(theta), float(delta))
    tau = _TAU_MEMO.get(key)
    if tau is None:
        tau = _bb8_tau_quad(*key)
        _remember(_TAU_MEMO, key, tau)
    if tau > 0.0:
        _remember(_THETA_MEMO, (tau, key[1]), key[0])
    return tau


def _theta_of(tau: float, delta: float, family_name: str) -> float:
    """θ with τ(θ, δ) = τ, by Brent on ``ln(θ − 1)``; clamped at ``_THETA_HI``.

    δ = 1 is Joe's own inversion (``_joe_theta_from_tau``), taken *before* the
    memo so the Joe member never depends on the call history and builds the
    very θ ``CopulaJoe`` builds.
    """
    tau = float(tau)
    if not (0.0 < tau < 1.0):
        raise CopulaParameterError(
            f'{family_name}: tau_k={tau!r} must be in (0, 1) — BB8 has no '
            f'negative dependence, and independence (τ = 0, θ = 1) is the '
            f'boundary of its admissible set, not a member of the range.')
    if delta >= 1.0:
        return min(_joe_theta_from_tau(tau), _THETA_HI)
    hit = _THETA_MEMO.get((tau, float(delta)))
    if hit is not None:
        return hit
    if tau >= _tau_of(_THETA_HI, delta):
        return _THETA_HI

    def f(s):
        return _tau_of(1.0 + math.exp(s), delta) - tau

    s_lo, s_hi = math.log(_THETA_LO - 1.0), math.log(_THETA_HI - 1.0)
    if f(s_lo) >= 0.0:
        return _THETA_LO
    s = brentq(f, s_lo, s_hi, xtol=1e-13, rtol=4.0 * np.finfo(float).eps, maxiter=200)
    theta = 1.0 + math.exp(s)
    _remember(_THETA_MEMO, (tau, float(delta)), theta)
    return theta


# ---------------------------------------------------------------------------
# Log-space kernel on ka = ln(1 − u), kb = ln(1 − v) (module docstring)
# ---------------------------------------------------------------------------

_LOG2: float = math.log(2.0)


def _log1p_neg_exp(x):
    """``ln(1 − e^{x})`` for x ≤ 0, degrading to ``x`` as x → −∞.

    ``log1p(−e^{x})`` underflows to 0.0 (whose log is −∞) once ``e^{x}``
    falls below the double's resolution of 1; below x = −36 the series
    ``ln(1 − y) = −y − y²/2 − …`` is used instead, whose next term is under
    10⁻³¹ there. Used only on the ``A ≤ ½`` branch of :func:`_bb8_terms_k`.
    """
    x = np.asarray(x, dtype=float)
    tiny = x < -36.0
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(tiny,
                        -np.exp(x) * (1.0 + 0.5 * np.exp(x)),
                        np.log1p(-np.exp(np.where(tiny, -1.0, x))))


def _bb8_log_p(ka, de):
    """``ln P``, ``P = 1 − δu = (1 − δ) + δ(1 − u)``, from ``ka = ln(1 − u)``.

    Two exact forms, switched at P = ½ (module docstring, "Numerics"):
    ``log1p(δ·expm1(ka))`` where P is not small, ``ka + ln(δ + (1−δ)e^{−ka})``
    where it is. At δ = 1 the second is ``ka`` exactly.
    """
    ka = np.asarray(ka, dtype=float)
    e = 1.0 - de
    x = de * np.expm1(ka)                      # = −δu ∈ [−δ, 0]
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        near = x > -0.5
        lo = ka + np.log(de + e * np.exp(-np.where(near, 0.0, ka)))
        return np.where(near, np.log1p(np.where(near, x, 0.0)), lo)


def _bb8_log_rho(ka, de):
    """``ln(P/E) = log1p(δ(1 − u)/E)`` — cancellation-free as u → 1.

    ``+∞`` at δ = 1 (E = 0), where ``P^θ − E^θ = P^θ`` and the ``−expm1(−θρ)``
    it feeds is exactly 1.
    """
    ka = np.asarray(ka, dtype=float)
    e = 1.0 - de
    if e <= 0.0:
        return np.full(np.shape(ka), math.inf)
    with np.errstate(over='ignore'):
        return np.log1p(de * np.exp(ka) / e)


def _bb8_terms_k(ka, kb, th, de):
    """``(lnP, lnQ, l1mQ, ln_eta, ln_a, ln1ma)`` — the log-space pieces of the
    module docstring, with ``ln A`` and ``ln(1 − A)`` each on the branch that
    computes it without cancellation.

    ``ln A = ln(1 − P^θ) + ln(1 − Q^θ) − ln η`` is exact where A is small —
    that is, towards (u, v) = (0, 0), where ``1 − P^θ ≈ θδu``.

    ``ln(1 − A) = ln D − ln η`` with
    ``ln D = logaddexp(ln(Q^θ(1 − P^θ)), ln(P^θ − E^θ))`` is exact where
    ``1 − A`` is small — towards (1, 1) — both arguments being logs of
    non-negative quantities there.

    Each is the *wrong* form at the other end, and by the whole value: at
    (u, v) = (10⁻¹², 10⁻¹²) the true ``ln(1 − A)`` is −10⁻²⁴ while ``ln D``
    and ``ln η`` are separately rounded numbers of order 1, so their
    difference is 0 and C comes out **exactly zero** instead of 10⁻²⁴. The
    switch is at ``A = ½``, where both forms are safe (``ln D − ln η ≤ −ln 2``
    on one side, ``A ≤ ½`` on the other), and ``log1p(−e^{lnA})`` degrades
    gracefully to ``−e^{lnA}`` for ``ln A`` below −36.
    """
    lnp = _bb8_log_p(ka, de)
    lnq = _bb8_log_p(kb, de)
    p = th * lnp
    q = th * lnq
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        l1mp = np.log(-np.expm1(p))
        l1mq = np.log(-np.expm1(q))
        e = 1.0 - de
        ln_eta = 0.0 if e <= 0.0 else float(np.log(-np.expm1(th * math.log1p(-de))))
        ln_a = l1mp + l1mq - ln_eta
        # ln(P^θ − E^θ) = θ ln P + ln(1 − (E/P)^θ)
        ln_pe = p + np.log(-np.expm1(-th * _bb8_log_rho(ka, de)))
        ln_d = np.logaddexp(q + l1mp, ln_pe)
        small = ln_a < -_LOG2
        ln1ma = np.where(small,
                         _log1p_neg_exp(np.where(small, ln_a, -1.0)),
                         ln_d - ln_eta)
    return lnp, lnq, l1mq, ln_eta, np.minimum(ln_a, 0.0), np.minimum(ln1ma, 0.0)


def _bb8_logpdf_k(ka, kb, th, de):
    """``ln c(u, v)`` from ``ka = ln(1 − u)``, ``kb = ln(1 − v)``."""
    lnp, lnq, _, ln_eta, ln_a, ln1ma = _bb8_terms_k(ka, kb, th, de)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        # ln(1 − A/θ) = logaddexp(ln(1 − A), ln A + ln(1 − 1/θ)); the second
        # term is −∞ at θ = 1 and absorbed exactly (independence: ln c ≡ 0).
        ln_b = np.logaddexp(ln1ma, ln_a + (math.log1p(-1.0 / th) if th > 1.0 else -math.inf))
        return (math.log(th) + math.log(de)
                - ln_eta
                + (th - 1.0) * (lnp + lnq)
                + (1.0 / th - 2.0) * ln1ma
                + ln_b)


def _bb8_cdf_k(ka, kb, th, de):
    """``(C, 1 − C)`` with ``C = (1 − (1 − A)^{1/θ})/δ``.

    The complement is ``1 − C = ((1 − A)^{1/θ} − E)/δ``, ``E = 1 − δ``, which
    cancels to nothing as (u, v) → (1, 1) — there ``1 − A = E^θ`` *exactly*
    (A = η there). It is therefore never formed as that difference but as

        ``1 − C = (E/δ)·expm1( ln(1 − A)/θ − ln E )``,

    whose argument is ``≥ 0`` on the whole square (``1 − A ≥ E^θ``, with
    equality only at the corner) and exactly 0 at it. At δ = 1, E = 0 and the
    expression degenerates: ``1 − C = (1 − A)^{1/θ}``, Joe's own complement,
    used directly.
    """
    ln1ma = _bb8_terms_k(ka, kb, th, de)[5]
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        ln_g = ln1ma / th                              # ln (1 − A)^{1/θ}
        c = -np.expm1(ln_g) / de
        if de >= 1.0:
            return c, np.exp(ln_g)
        log_e = math.log1p(-de)
        return c, ((1.0 - de) / de) * np.expm1(np.maximum(ln_g - log_e, 0.0))


def _bb8_logh_k(kb, ka, th, de):
    """``ln h(v|u) = (θ − 1) ln P + ln(1 − Q^θ) − ln η + (1/θ − 1) ln(1 − A)``."""
    lnp, _, l1mq, ln_eta, _, ln1ma = _bb8_terms_k(ka, kb, th, de)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        return ((th - 1.0) * lnp
                + l1mq
                - ln_eta
                + (1.0 / th - 1.0) * ln1ma)


def _bb8_h_k(kb, ka, th, de):
    """``(h, 1 − h)`` of h(v|u), through :func:`_bb8_logh_k`."""
    logh = _bb8_logh_k(kb, ka, th, de)
    return np.exp(logh), -np.expm1(logh)


def _bb8_inv_h_k(lw, ka, th, de):
    """``(v, 1 − v)`` solving h(v|u) = w, from ``lw = ln w``, ``ka = ln(1 − u)``.

    Vectorised bisection on ``x = logit v`` (module docstring): ln h increases
    with v hence with x, and ``kb = ln(1 − v) = −logaddexp(0, x)`` is exact,
    so 1 − v is never formed in linear scale. ``w`` outside
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
        below = lw <= _bb8_logh_k(math.log1p(-EPS), ka, th, de)
        above = lw >= _bb8_logh_k(math.log(EPS), ka, th, de)
        for _ in range(_BISECT_STEPS):
            mid = 0.5 * (lo + hi)
            kb = -np.logaddexp(0.0, mid)            # ln(1 − expit(mid))
            up = _bb8_logh_k(kb, ka, th, de) < lw
            lo = np.where(up, mid, lo)
            hi = np.where(up, hi, mid)
        x = 0.5 * (lo + hi)
        x = np.where(below, x_lo, np.where(above, -x_lo, x))
        kb = -np.logaddexp(0.0, x)
    return -np.expm1(kb), np.exp(kb)


# ---------------------------------------------------------------------------
# Copula class
# ---------------------------------------------------------------------------

class CopulaBB8(CopulaVirt):
    """BB8 (Joe-Frank) copula — two parameters, no tail dependence for δ < 1.

    Parameters
    ----------
    tau_k : float
        Kendall's τ ∈ (0, 1). Every value of the registered range is reachable
        at every δ — BB8 has **no** joint (τ, δ) constraint (module
        docstring) — but a τ above ``τ(θ = 10⁷, δ)`` builds at that θ cap and
        the copula then stores the τ it realises.
    delta8 : float, optional
        δ ∈ (0, 1], default 1.0 (the **Joe** member). Smaller δ moves the
        family towards Frank (a limit reached only as θ → ∞ with θδ fixed) and
        removes upper-tail dependence outright: λ_U = 0 for every δ < 1.
        Named ``delta8``, not ``delta`` (BB1) or ``delta6`` (BB6), so that the
        joint fitter cannot route BB8 through another family's branch (module
        docstring).
    """

    n_params: int = 2

    def __init__(self, **kwargs):
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    @classmethod
    def reachable_tau_bounds(cls) -> tuple[float, float]:
        """``(EPS, τ(θ = 10⁷, δ = 1))`` — the largest cap over δ, since τ
        increases with δ at fixed θ (Tawn's ψ = 1 precedent)."""
        return float(EPS), _tau_of(_THETA_HI, 1.0)

    @classmethod
    def _delta_error(cls, delta: float) -> str | None:
        """Why the constructor refuses δ, or ``None`` when it accepts.

        δ alone decides: unlike BB1, BB6 and Tawn, BB8 puts no joint
        constraint on (τ, δ) (module docstring, "Parametrisation").
        """
        if not np.isfinite(delta) or not 0.0 < delta <= 1.0:
            return f'BB8: delta8 must be in (0, 1], got {delta!r}.'
        return None

    @classmethod
    def constructible_params(cls, params: dict) -> dict:
        """``params`` itself when BB8 builds at it; else δ moved into (0, 1].

        Only δ can be wrong here — there is no (τ, δ) pair the constructor
        refuses for τ's sake — so a τ the registry lets through is never
        touched. A δ that is not a number at all is returned unchanged, for
        the constructor to refuse.
        """
        delta = params.get("delta8", _DELTA_DEFAULT)
        if isinstance(delta, (bool, np.bool_)):          # the constructor refuses it
            return params
        try:
            delta_f = float(delta)
        except (TypeError, ValueError):
            return params
        if cls._delta_error(delta_f) is None:
            return params
        if not np.isfinite(delta_f):
            return params
        lo, _, _ = cls._delta_box()
        return {**params, "delta8": float(min(max(delta_f, lo), 1.0))}

    @staticmethod
    def _delta_box() -> tuple[float, float, float]:
        from pmcprg.copulas._base import EXTRA_PARAM_BOUNDS_BY_PARAM
        return EXTRA_PARAM_BOUNDS_BY_PARAM["delta8"]

    @classmethod
    def constrain_params(cls, params: dict) -> dict:
        """δ clipped into ``EXTRA_PARAM_BOUNDS_BY_PARAM['delta8']`` — the whole
        constraint, the box *being* the admissible set here."""
        lo, hi, init = cls._delta_box()
        out = dict(params)
        out["delta8"] = float(np.clip(float(out.get("delta8", init)), lo, hi))
        return out

    def _update_params(self):
        tau = float(self.params['tau_k'])
        delta = self.params.get('delta8', _DELTA_DEFAULT)
        if isinstance(delta, (bool, np.bool_)):
            raise CopulaParameterError(f'BB8: delta8={delta!r} is not a number.')
        try:
            delta = float(delta)
        except (TypeError, ValueError):
            raise CopulaParameterError(f'BB8: delta8={delta!r} is not a number.') from None
        err = self._delta_error(delta)
        if err is not None:
            raise CopulaParameterError(err)
        theta = _theta_of(tau, delta, 'BB8')
        self.theta = theta
        self.delta8 = delta
        self.params['delta8'] = delta
        if theta >= _THETA_HI:
            # θ clamped: store the τ it realises (RB-10, as Galambos and Tawn).
            self.params['tau_k'] = _tau_of(theta, delta)

    def tau_of(self) -> float:
        """τ recomputed from (θ, δ) — the forward map of the constructor."""
        return _tau_of(self.theta, self.delta8)

    # ------------------------------------------------------------------
    # Kernel interface — Joe's coordinates (module docstring)
    # ------------------------------------------------------------------

    @staticmethod
    def _kcoord(x):
        return np.log1p(-x)

    def _k_logpdf(self, ka, kb):
        return _bb8_logpdf_k(ka, kb, self.theta, self.delta8)

    def _k_cdf(self, ka, kb):
        return _bb8_cdf_k(ka, kb, self.theta, self.delta8)

    def _k_h(self, kb, ka):
        return _bb8_h_k(kb, ka, self.theta, self.delta8)

    def _k_inv_h(self, lw, ka):
        return _bb8_inv_h_k(lw, ka, self.theta, self.delta8)

    # ------------------------------------------------------------------
    # CDF / PDF / h-function
    # ------------------------------------------------------------------

    def cdf(self, uv):
        """C(u,v) = (1 − (1 − A)^{1/θ})/δ (module docstring)."""
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
        """c(u,v) = θδ(PQ)^{θ−1}(1 − A)^{1/θ−2}(1 − A/θ)/η (module docstring).

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
        """h(v|u) = P^{θ−1}(1 − Q^θ)(1 − A)^{1/θ−1}/η."""
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
        """λ_L = 0 always; λ_U = 0 for δ < 1 and 2 − 2^{1/θ} at δ = 1.

        The discontinuity is the family's, not an approximation: for δ < 1,
        ``1 − φ^{-1}(s)`` leaves 0 with a finite non-zero slope, which sends
        the upper-tail ratio to 2 (module docstring).
        """
        if self.delta8 >= 1.0:
            return 0.0, float(2.0 - 2.0 ** (1.0 / self.theta))
        return 0.0, 0.0

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    @classmethod
    def fit(cls, data: np.ndarray, method: str = 'mle'):
        """Joint MLE of (τ, δ); ``method='tau'`` falls back to it.

        Kendall's τ alone cannot identify δ — τ(θ, δ) has two unknowns — so a
        ``'tau'`` request logs a warning and is answered by the same joint
        two-parameter MLE as ``'mle'`` (BB1's, BB6's and Tawn's precedent).
        """
        if method == 'tau':
            logger.warning(
                "%s.fit: method='tau' cannot identify delta8 from Kendall's tau "
                "alone; falling back to MLE.", cls.__name__)
            method = 'mle'
        return super().fit(data, method=method)


if __name__ == '__main__':
    from pathlib import Path

    from pmcprg.copulas.archimedean.joe import CopulaJoe
    from pmcprg.copulas.explicit.product import CopulaProduct

    cop = CopulaBB8(tau_k=0.5, delta8=0.6)
    lL, lU = cop.tail_dependence()
    print(f'Copula : {cop.copula_enum.value.LONG_NAME}')
    print(f'tau_k  : {cop.params["tau_k"]:.4f}  delta8={cop.delta8:.4f}  theta={cop.theta:.6f}')
    print(f'tau    check   : {cop.tau_of():.15f}  (want {cop.params["tau_k"]:.15f})')
    print(f'pdf(0.3, 0.7) = {cop.pdf([0.3, 0.7]):.6f}')
    print(f'cdf(0.3, 0.7) = {cop.cdf([0.3, 0.7]):.6f}')
    print(f'h(0.7 | 0.3)  = {cop.conditional_cdf(0.7, 0.3):.6f}')
    print(f'tail dep : λ_L = {lL:.4f},  λ_U = {lU:.4f}  [0 for every delta8 < 1]')

    print('\nδ = 1 vs Joe @ (0.3, 0.7), τ = 0.6:')
    print(f'  BB8.pdf={CopulaBB8(tau_k=0.6, delta8=1.0).pdf([0.3, 0.7]):.12f}  '
          f'Joe.pdf={CopulaJoe(tau_k=0.6).pdf([0.3, 0.7]):.12f}')
    print('θ = 1 (τ → 0) vs Product @ (0.3, 0.7), δ = 0.5:')
    print(f'  BB8.pdf={CopulaBB8(tau_k=1e-14, delta8=0.5).pdf([0.3, 0.7]):.12f}  '
          f'Prod.pdf={CopulaProduct(tau_k=0.0).pdf([0.3, 0.7]):.12f}')

    plot_dir = Path('./data/Plots/Copulas')
    plot_dir.mkdir(parents=True, exist_ok=True)
    cop.plot_pdf(plot_dir)
    cop.plot_cdf(plot_dir)
    cop.plot_h_function(plot_dir)
    cop.plot_samples(plot_dir)
    cop.plot_overview(plot_dir)
