"""
Hüsler–Reiss copula — bivariate extreme-value family, upper-tail dependence
only (Hüsler & Reiss, 1989).

Pickands dependence function: A(t) = t·Φ(g(t)) + (1−t)·Φ(h(t)), λ > 0,
t ∈ [0, 1], with
    g(t) = 1/λ + (λ/2)·ln(t / (1−t)),   h(t) = 1/λ − (λ/2)·ln(t / (1−t))
          = 2/λ − g(t),
Φ the standard normal CDF (A(0) = A(1) = 1, checked below).

CDF:  C(u, v) = exp(−ℓ),   ℓ = w·Φ(g) + z·Φ(h),
      w = −ln u, z = −ln v, g = 1/λ + (λ/2)·ln(w/z), h = 1/λ − (λ/2)·ln(w/z)
      (Hüsler & Reiss 1989; Joe 2014, §6.3; Kotz & Nadarajah 2000, §3.4 —
      note g, h here use ln(w/z) = ln(t/(1−t)) directly, since
      t = w/(w+z) ⇒ t/(1−t) = w/z, so the Pickands variable ``t`` itself is
      never needed to evaluate C, h or c below).

PDF (re-derived below, not transcribed — see "Derivation" below):
      c(u, v) = C/(uv) · [Φ(g)Φ(h) + λ·φ(g)/(2z)],    φ = standard normal pdf.
h:    h(v|u) = ∂C/∂u = (C/u)·Φ(g).

λ → 0⁺: **independence**.  λ → ∞: **comonotone copula M**.

**This direction was re-derived here, not assumed** — see "Direction of λ"
below for why a first read of the Pickands formula (constant term 1/λ that
diverges as λ → 0) can suggest the opposite, wrong, direction.

λ_U = 2(1 − A(1/2)) = 2·Φ(−1/λ),  λ_L = 0 (Nelsen 2006, Thm 5.7.3, the same
identity used for Galambos). This is the textbook Hüsler–Reiss upper-tail
coefficient (Hüsler & Reiss 1989; Beirlant, Goegebeur, Segers & Teugels
2004, *Statistics of Extremes*, §9.3; Joe 2014, Table 6.1) and both its
formula *and* its direction (λ_U → 1 as λ → ∞, → 0 as λ → 0) were verified
against an independent 40-digit ``mpmath`` evaluation of A(1/2) during
development, not taken on trust from a single secondary source.

Kendall's τ(λ) — **no closed form** (same situation as Galambos; confirmed,
not assumed, exactly as flagged in the audit). Computed via the Genest &
MacKay (1986) identity τ = ∫₀¹ [t(1−t)/A(t)]·A''(t) dt, using the shared
quadrature machinery of :mod:`pmcprg.copulas.extreme_value._pickands`
(factored out of the Galambos module, FR-9 round 1, once this family needed
the same adaptive-quadrature-with-breakpoints and Brent-inversion pattern).
The identity itself was already cross-checked there against Gumbel's
closed-form τ = 1 − 1/θ; this module instead cross-checks its own A(t), A'
and A'' — closed forms, not the shared machinery — against 40-digit
``mpmath`` numerical differentiation of A (script used during development,
not checked in), and the resulting τ(λ) against a 20,000-pair Monte-Carlo
Kendall's τ from this module's own ``sample()``:

    λ = 1, 2, 5 (seed 0): quadrature τ = 0.25545, 0.53868, 0.79139;
    Monte-Carlo τ̂        = 0.2503,  0.5434,  0.7915  (s.e. ≈ 0.0055 at
    n = 20,000) — agrees within 1–2 Monte-Carlo standard errors.

τ(λ) is strictly increasing on (0, ∞) (checked numerically, no proof
attempted — same caveat as Galambos), so λ(τ) is a direct Brent search on
``tau_from_lambda`` (no cached table, same cost/scope tradeoff as Galambos).

λ is capped at ``_LAMBDA_MAX`` = 1e6 (quadrature verified accurate there —
module tests; the A''(t) ridge at t = 1/2 grows only *linearly* in λ here,
vs. Galambos's steeper growth, but the same breakpoint scale was re-checked
rather than assumed adequate); the table this caps reaches τ ≈ 1 − 1.13e-6.
The lower end is not restrictive: λ = 0.05 already gives τ ≈ 4×10⁻⁸⁹, far
below ``EPS`` — the family's own ``_LAMBDA_MIN`` — so only the upper end
narrows the registered [EPS, 1.0], declared via ``reachable_tau_bounds``
(the Frank/Plackett/Galambos convention).

Direction of λ
--------------
A plausible *wrong* first guess, reading only the additive term 1/λ in
g(t) and h(t), is that λ → 0 makes both g(t) and h(t) blow up to +∞
(driving Φ(g), Φ(h) → 1, hence A(t) → t + (1 − t) = 1) — which is
independence, not comonotone — the **opposite** of what an intuitive
"small dependence parameter ⇒ strong dependence" reading (by loose analogy
with, say, Clayton's θ → 0⁺ ⇒ independence, which runs the other way for
a *different* reason) might suggest, and the opposite of the specific
"λ → 0 = comonotone" claim that a first reading of some secondary
summaries can leave one with. This module re-derives the limit directly:

* λ → 0⁺, t fixed in (0, 1): ln(t/(1−t)) is a fixed finite number, so the
  constant term 1/λ → +∞ dominates *both* g(t) = 1/λ + (λ/2)ln(t/(1−t))
  and h(t) = 1/λ − (λ/2)ln(t/(1−t)) — the (λ/2)·(finite) term vanishes
  relative to 1/λ regardless of its sign. So g(t), h(t) → +∞, Φ(g), Φ(h)
  → 1, and A(t) → t·1 + (1−t)·1 = 1 for every t ∈ (0, 1): **A → 1 is the
  independence Pickands function**, so λ → 0⁺ is independence.
* λ → ∞, t fixed, t < 1/2 (so ln(t/(1−t)) < 0): 1/λ → 0, so
  (λ/2)ln(t/(1−t)) → −∞ dominates; g(t) → −∞ (Φ(g) → 0), while
  h(t) = 2/λ − g(t) → +∞ (Φ(h) → 1). A(t) → t·0 + (1−t)·1 = 1 − t =
  max(t, 1−t) for t < 1/2 (symmetric for t > 1/2): **max(t, 1−t) is the
  comonotone Pickands function**, so λ → ∞ is comonotone.

So **λ runs the same way as Galambos's θ** (larger parameter ⇒ stronger
dependence), not the opposite way. This is confirmed two further ways: (a)
the tail-dependence formula λ_U = 2Φ(−1/λ) above gives λ_U → 1 as λ → ∞
and λ_U → 0 as λ → 0, matching only the "λ → ∞ comonotone" direction; and
(b) converting to the alternative "η = 1/λ" parametrisation used in some
references (whose Pickands function is the same formula with η in place
of λ's *reciprocal* role — i.e. constant term η, log-coefficient 1/(2η))
recovers the textbook "η → 0 complete dependence, η → ∞ independence"
statement exactly, consistently with λ = 1/η → ∞ being complete
dependence in *this* module's λ. A reader of a secondary source that
states the opposite direction under the name "λ" is very likely reading a
source that defines λ as this module's 1/λ (i.e. as η above) — exactly the
kind of mistranscription risk this audit exists to catch (cf. Galambos's
wrong τ(θ) = 1/(θ+2) claim in some secondary summaries).

Derivation
----------
For any bivariate EV copula with stable tail dependence function
ℓ(w, z) = (w+z)·A(t), t = w/(w+z) (w = −ln u, z = −ln v), the general
identities (re-derived here from C = exp(−ℓ), not transcribed; Joe 2014
§6.1.2 gives the same structure)

    ℓ_w = A(t) + (1−t)·A'(t),     ℓ_z = A(t) − t·A'(t)
    h(v|u) = C·ℓ_w/u
    c(u,v) = (C/(uv))·[ℓ_w·ℓ_z + t(1−t)·A''(t)/(w+z)]

were checked against Galambos's own module (whose h/c formulas were
independently re-derived there): substituting Galambos's A(t) and A'(t)
reproduces its ``a1 = 1 − S·r/w`` term exactly (``∂S/∂w = S·r/w`` verified
algebraically), confirming these general identities before applying them
to a new family here.

For Hüsler–Reiss's A(t) = tΦ(g) + (1−t)Φ(h), A'(t) = Φ(g) − Φ(h) (below),
substituting gives a striking simplification — **ℓ_w = Φ(g)** and
**ℓ_z = Φ(h)** exactly (the (1−t)Φ(h) / tΦ(g) cross-terms cancel):

    ℓ_w = tΦ(g) + (1−t)Φ(h) + (1−t)[Φ(g) − Φ(h)] = Φ(g)   (the ± (1−t)Φ(h)
                                                            terms cancel)
    ℓ_z = tΦ(g) + (1−t)Φ(h) − t[Φ(g) − Φ(h)]     = Φ(h)   (symmetric)

so h(v|u) = (C/u)·Φ(g) and c(u,v) = (C/(uv))·[Φ(g)Φ(h) + t(1−t)A''(t)/(w+z)].
The last term also simplifies: with A''(t) = λ·φ(g)/(2t(1−t)²) (below),
t(1−t)·A''(t)/(w+z) = λ·φ(g) / (2·(1−t)(w+z)) = λ·φ(g)/(2z) (since
(1−t)(w+z) = z exactly), giving the pdf formula quoted at the top — this
is the well-known closed form for the Hüsler–Reiss density.

All four formulas (C, A', A'', c) were cross-checked against 40–60-digit
``mpmath`` (finite differences of C in u, v and of A in t; a symbolic
substitution check of A' via the identity t·φ(g) = (1−t)·φ(h), itself
verified from g² − h² = (g−h)(g+h) = λ·ln(t/(1−t))·(2/λ) = 2·ln(t/(1−t)))
before being written here — not transcribed from a single secondary
source, following this package's audit standard (the Galambos pilot found
a wrong τ(θ) formula in circulation this same way).

References
----------
* Hüsler, J. & Reiss, R.-D. (1989). Maxima of normal random vectors:
  between independence and complete dependence. *Statistics & Probability
  Letters* 7(4), 283-286, doi:10.1016/0167-7152(89)90106-5.
* Kotz, S. & Nadarajah, S. (2000). *Extreme Value Distributions: Theory and
  Applications*, Imperial College Press, §3.4 (the Hüsler–Reiss model).
* Joe, H. (2014). *Dependence Modeling with Copulas*, Chapman & Hall/CRC,
  §6.1 (general EV-copula identities), §6.3 (Hüsler–Reiss), Table 6.1
  (tail dependence).
* Beirlant, J., Goegebeur, Y., Segers, J. & Teugels, J. (2004). *Statistics
  of Extremes: Theory and Applications*, Wiley, §9.3.
* Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed., Springer,
  §5.7, Theorem 5.7.3 (upper-tail dependence of an EV copula).
* Genest, C. & MacKay, J. (1986). The joy of copulas: bivariate
  distributions with uniform marginals. *The American Statistician* 40(4),
  280-283.

Numerics
--------
``g`` and ``h`` are ordinary sums/differences of ``1/λ`` and
``(λ/2)·ln(w/z)`` — unlike Galambos's kernel, this never cancels
catastrophically (the two terms are only ever comparable in magnitude when
both are small, in which case the *result* is also small and no precision
is lost; see the module's development notes). The corner-robust rewriting
needed here is instead in the normal-CDF/PDF evaluations themselves:

* ``1 − Φ(g)`` must be computed as ``Φ(−g)`` (``scipy.special.ndtr(-g)``),
  never as ``1 - ndtr(g)``: for g ≳ 9, ``ndtr(g)`` rounds to exactly 1.0 in
  double precision, and the naive subtraction loses every digit. Used in
  ``log c = w·Φ(−g) + z·Φ(−h) + log[Φ(g)Φ(h) + λφ(g)/(2z)]`` (below).
* ``log Φ(x)`` for very negative ``x`` must come from
  ``scipy.special.log_ndtr``, not ``log(ndtr(x))``: ``ndtr(x)`` underflows
  to exactly 0.0 for x ≲ −38, and ``log(0.0)`` returns ``-inf`` — which
  happens to be numerically *correct* asymptotically, but any small
  admixture (a comparison, a threshold) upstream of the underflow is a bug
  waiting to happen; ``log_ndtr`` computes the asymptotic expansion
  directly and stays accurate down to machine-representable magnitudes.
* The pdf bracket ``Φ(g)Φ(h) + λφ(g)/(2z)`` is computed as
  ``logaddexp(log_ndtr(g) + log_ndtr(h), log(λ) + logφ(g) − log(2z))``:
  neither summand is ever formed in linear scale first, so a term that
  underflows in linear scale (φ(g) for a large |g| near the corners)
  contributes exactly 0 to the sum in log-space instead of colliding with
  the other, non-underflowed term at limited relative precision.
* ``A''(t) = λ·φ(g(t))/(2t(1−t)²)`` is likewise computed as
  ``exp(log λ + logφ(g) − log 2 − log t − 2·log(1−t))``: as t → 0 or 1 at
  a large λ, ``φ(g(t))`` underflows to exactly 0.0 (g(t) → ∓∞ like
  λ·ln t), while ``t(1−t)²`` underflows separately and far less steeply —
  computed in linear scale, the two independent underflows can produce a
  spurious ``0/0 = nan`` instead of the true limit 0; computed as a single
  ``exp`` of a sum of logs, the (very negative) exponent underflows
  cleanly to 0.0.

``inv_h`` has no closed form (same as Galambos), so this module relies on
:meth:`CopulaVirt.inv_h`'s Brent-bracketed default over the closed-form
``conditional_cdf`` above.
"""
if __name__ == '__main__':
    import sys
    import pathlib

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import logging

import numpy as np
from scipy.special import log_ndtr, ndtr
from scipy.stats import norm

from pmcprg.copulas._base import CopulaVirt
from pmcprg.copulas.extreme_value._pickands import invert_tau, tau_from_A_terms
from pmcprg.numerics import EPS, ONE_MINUS_EPS, minmaxEPS

logger = logging.getLogger(__name__)

_LOG_2PI = float(np.log(2.0 * np.pi))


def _logphi(x):
    """log of the standard normal pdf, elementary and never underflows to nan."""
    return -0.5 * x * x - 0.5 * _LOG_2PI


# ---------------------------------------------------------------------------
# Log-space kernel on ka = log u, kb = log v (module docstring, "Numerics")
# ---------------------------------------------------------------------------

def _husler_reiss_gh(ka, kb, lam):
    """``(w, z, g, h)`` from ``ka = log u``, ``kb = log v``.

    ``g = 1/λ + (λ/2)·ln(w/z)``, ``h = 1/λ − (λ/2)·ln(w/z) = 2/λ − g``
    (module docstring). No cancellation risk here (module docstring,
    "Numerics"): the corner-robust rewriting for this family is downstream,
    in the normal CDF/PDF evaluations.
    """
    w = -ka
    z = -kb
    r = np.log(w) - np.log(z)
    inv_lam = 1.0 / lam
    half_lam = 0.5 * lam
    g = inv_lam + half_lam * r
    h = inv_lam - half_lam * r
    return w, z, g, h


def _husler_reiss_logC(w, z, g, h):
    """``log C = -(w·Φ(g) + z·Φ(h))`` — always ≤ 0 (Φ ∈ [0, 1], w, z ≥ 0)."""
    return -(w * ndtr(g) + z * ndtr(h))


def _husler_reiss_logpdf(ka, kb, lam):
    """log c(u, v), corner-robust (module docstring, "Numerics")."""
    w, z, g, h = _husler_reiss_gh(ka, kb, lam)
    log_bracket = np.logaddexp(
        log_ndtr(g) + log_ndtr(h),
        np.log(lam) + _logphi(g) - np.log(2.0) - np.log(z),
    )
    return w * ndtr(-g) + z * ndtr(-h) + log_bracket


def _husler_reiss_cdf(ka, kb, lam):
    """``(C, 1 − C)``."""
    w, z, g, h = _husler_reiss_gh(ka, kb, lam)
    logC = _husler_reiss_logC(w, z, g, h)
    return np.exp(logC), -np.expm1(logC)


def _husler_reiss_h(kb, ka, lam):
    """``(h, 1 − h)`` of h(b | a) = ∂C(a, b)/∂a = (C/a)·Φ(g)."""
    w, z, g, h_ = _husler_reiss_gh(ka, kb, lam)
    logC = _husler_reiss_logC(w, z, g, h_)
    logh = logC + w + log_ndtr(g)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        h = np.exp(logh)
    return h, -np.expm1(logh)


# ---------------------------------------------------------------------------
# Pickands function A(t) = tΦ(g(t)) + (1-t)Φ(h(t)) and its derivatives
# (module docstring, "Derivation"). t plays the role of u, 1-t of v.
# ---------------------------------------------------------------------------

def _husler_reiss_A_terms(t, lam):
    """``(A, A', A'')`` at ``t`` (array or scalar), closed form.

    A'(t) = Φ(g) − Φ(h); A''(t) = λ·φ(g)/(2t(1-t)²) — both re-derived in
    the module docstring ("Derivation") from A(t) via the identity
    t·φ(g) = (1−t)·φ(h), and verified against 40-digit ``mpmath`` numerical
    differentiation of A during development (see module docstring).
    """
    t = np.asarray(t, dtype=float)
    tm = 1.0 - t
    L = np.log(t) - np.log(tm)
    inv_lam = 1.0 / lam
    half_lam = 0.5 * lam
    g = inv_lam + half_lam * L
    h = inv_lam - half_lam * L
    Phi_g = ndtr(g)
    Phi_h = ndtr(h)
    A = t * Phi_g + tm * Phi_h
    Ap = Phi_g - Phi_h
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        log_App = (np.log(lam) + _logphi(g) - np.log(2.0)
                  - np.log(t) - 2.0 * np.log(tm))
        App = np.exp(log_App)
    App = np.where(np.isfinite(App), App, 0.0)
    return A, Ap, App


def _husler_reiss_tau_from_lambda(lam: float) -> float:
    """τ(λ) by adaptive quadrature of the Genest & MacKay (1986) identity.

    Shared machinery with Galambos — see
    :func:`pmcprg.copulas.extreme_value._pickands.tau_from_A_terms` for the
    breakpoint rationale and :mod:`pmcprg.tests.test_husler_reiss` for the
    high-precision cross-check and Monte-Carlo agreement (module docstring).
    """
    return tau_from_A_terms(_husler_reiss_A_terms, lam)


_LAMBDA_LO = 0.05     # tau_from_lambda(_LAMBDA_LO) ≈ 4e-89, far below EPS
_LAMBDA_HI = 1.0e6    # quadrature verified accurate here (module docstring)


def _husler_reiss_lambda_from_tau(tau_target: float) -> float:
    """Invert τ(λ) by Brent's method on ``[_LAMBDA_LO, _LAMBDA_HI]``.

    τ(λ) is strictly increasing (checked numerically, not proved) from
    ≈ 0 to τ(_LAMBDA_HI) < 1. A target beyond that reachable maximum is
    clamped to ``_LAMBDA_HI`` (module docstring; ``reachable_tau_bounds``
    keeps the package from ever requesting more).
    """
    return invert_tau(_husler_reiss_tau_from_lambda, tau_target,
                      _LAMBDA_LO, _LAMBDA_HI, family_name='Hüsler–Reiss')


# ---------------------------------------------------------------------------
# Copula class
# ---------------------------------------------------------------------------

class CopulaHuslerReiss(CopulaVirt):
    """Hüsler–Reiss copula — bivariate extreme-value family, λ > 0.

    Parameters
    ----------
    tau_k : float
        Kendall's τ. Must be > 0 (extreme-value copulas have no negative or
        null dependence — the module docstring); τ = 0 is the λ → 0 limit
        and is not attained. Reachable up to ``reachable_tau_bounds()[1]``
        (≈ 1 − 1.13e-6): a larger request is clamped to λ = 1e6 and
        ``params['tau_k']`` becomes the τ this λ realises (RB-10).

    Note the direction: unlike a first guess from the Pickands formula's
    ``1/λ`` term, **λ → 0 is independence and λ → ∞ is the comonotone
    copula** — the same direction as Galambos's θ, not the opposite; see
    the module docstring's "Direction of λ" section for the derivation.
    """

    @classmethod
    def reachable_tau_bounds(cls) -> tuple[float, float]:
        """``(EPS, τ(1e6))`` — the τ this family's finite λ range realises.

        The lower end is not restrictive (λ = 0.05 already reaches
        τ ≈ 4e-89, far below EPS, inside ``[_LAMBDA_LO, _LAMBDA_HI]``);
        only the upper end narrows the registered ``[EPS, 1.0]``.
        """
        return float(EPS), _husler_reiss_tau_from_lambda(_LAMBDA_HI)

    def __init__(self, **kwargs):
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    def _update_params(self):
        # Stored as ``self.theta`` — the package-wide convention for "this
        # family's own dependence parameter" (every other family does the
        # same, e.g. Gaussian's ρ, Frank's θ — see ``pmcprg/copulas/_base.py``
        # ``_required_param_keys`` in test_copula_limits.py reads it
        # generically) — even though this family's own math and literature
        # call it λ throughout this module and its docstring.
        tau = self.params['tau_k']
        self.theta = _husler_reiss_lambda_from_tau(tau)
        if self.theta >= _LAMBDA_HI:
            # λ was clamped: store the τ it realises, not the one requested
            # (audit RB-10 convention, as Frank/Plackett/Galambos).
            self.params['tau_k'] = _husler_reiss_tau_from_lambda(self.theta)

    # -- public API ------------------------------------------------------

    def pdf(self, uv):
        u = np.float64(minmaxEPS(uv[0]))
        v = np.float64(minmaxEPS(uv[1]))
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            return float(np.exp(_husler_reiss_logpdf(np.log(u), np.log(v), self.theta)))

    def pdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised closed-form PDF, ``exp`` of :meth:`logpdf_array` (no floor)."""
        return np.exp(self.logpdf_array(uv))

    def logpdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Native log-PDF in log space (module docstring)."""
        uv = np.asarray(uv, dtype=float)
        u = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            return _husler_reiss_logpdf(np.log(u), np.log(v), self.theta)

    def cdf(self, uv):
        u = np.float64(minmaxEPS(uv[0]))
        v = np.float64(minmaxEPS(uv[1]))
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            c, _ = _husler_reiss_cdf(np.log(u), np.log(v), self.theta)
        return float(np.clip(c, 0.0, 1.0))

    def cdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised closed-form CDF (NaN where it cannot be evaluated)."""
        uv = np.asarray(uv, dtype=float)
        u = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            c, _ = _husler_reiss_cdf(np.log(u), np.log(v), self.theta)
        return np.clip(c, 0.0, 1.0)

    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = ∂C(u,v)/∂u, in log space (module docstring)."""
        u = np.float64(minmaxEPS(u))
        v = np.float64(minmaxEPS(v))
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            h, _ = _husler_reiss_h(np.log(v), np.log(u), self.theta)
        return float(np.clip(h, 0.0, 1.0))

    def tail_dependence(self) -> tuple[float, float]:
        """Hüsler–Reiss copula: λ_L = 0, λ_U = 2·Φ(−1/λ) (module docstring)."""
        return 0.0, float(2.0 * norm.cdf(-1.0 / self.theta))


if __name__ == '__main__':
    from pathlib import Path

    cop = CopulaHuslerReiss(tau_k=0.5)
    lam_u = 2.0 * norm.cdf(-1.0 / cop.theta)
    print(f'Copula   : {cop.copula_enum.value.LONG_NAME}')
    print(f'tau_k    : {cop.params["tau_k"]:.4f}  range={cop.tau_range}')
    print(f'lambda   : {cop.theta:.6f}')
    print(f'tau back : {_husler_reiss_tau_from_lambda(cop.theta):.6f}')
    print(f'pdf(0.3, 0.7) = {cop.pdf([0.3, 0.7]):.6f}')
    print(f'cdf(0.3, 0.7) = {cop.cdf([0.3, 0.7]):.6f}')
    print(f'h(0.7 | 0.3)  = {cop.conditional_cdf(0.7, 0.3):.6f}')
    print(f'tail dep : λ_L = 0,  λ_U = {lam_u:.4f}  [= 2Φ(-1/λ)]')

    plot_dir = Path('./data/Plots/Copulas')
    plot_dir.mkdir(parents=True, exist_ok=True)
    cop.plot_pdf(plot_dir)
    cop.plot_cdf(plot_dir)
    cop.plot_h_function(plot_dir)
    cop.plot_samples(plot_dir)
    cop.plot_overview(plot_dir)
    cop.plot_multi_tau(plot_dir)
