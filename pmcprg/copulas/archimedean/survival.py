"""
Survival (180° rotation) copulas.

Given a base copula C, its survival copula is:
    Ĉ(u,v) = u + v − 1 + C(1−u, 1−v)
    ĉ(u,v) = c(1−u, 1−v)
    ĥ(v|u) = 1 − h_C(1−v | 1−u)
    (λ_L, λ_U)  of Ĉ = (λ_U, λ_L) of C   [tail roles swap]
    τ_K is preserved (concordance invariant under (u,v)→(1−u,1−v)).

Concrete subclasses: SurvivalClayton, SurvivalGH, SurvivalJoe, SurvivalBB1
(FR-8, last round — the audit's own list of families needing 90°/270°
rotations also named "BB1 de survie", the 180° rotation of BB1, which this
module had not built yet even though its three one-parameter siblings were
already here).

SurvivalBB1's second parameter (``delta``) needs no special-casing in
:meth:`SurvivalCopula._update_params`: it already copies the *whole*
``self.params`` dict onto the base copula (``self._base.params =
self.params``), not a hand-picked subset keyed on ``tau_k`` alone — unlike
:class:`pmcprg.copulas.archimedean.rotated.RotatedCopula`, which negates
``tau_k`` and so has to rebuild the dict field by field. ``delta`` rides
along for free, exactly as ``tau_k`` already did for the three one-parameter
survivals.  What *is* new — BB1 is the first base family passed to
:class:`SurvivalCopula` whose ``constrain_params``/``constructible_params``
are not the identity (``δ < 1/(1 − τ)``, module docstring of ``bb1.py``) —
is the delegation added below: :meth:`SurvivalCopula.constrain_params` and
:meth:`SurvivalCopula.constructible_params` now forward to
``_base_class``'s own hooks unchanged (no sign flip: a 180° rotation leaves
τ's sign alone, unlike the 90°/270° case), so a multistart draw or a bounded
optimiser fitting ``SurvivalBB1`` sees the same admissible set as fitting
``CopulaBB1`` directly. Clayton/GH/Joe's own hooks are the identity, so
nothing changes for them (dict round-trips to the same values).

τ_K is unchanged under this 180° rotation (module docstring above), so
``SurvivalBB1.TAU_MIN_MAX`` mirrors ``CopulaBB1``'s own registered range
``[0 + ε, 1)`` bit for bit — exactly as ``SurvivalClayton``'s range mirrors
``CopulaClayton``'s ``[0 + ε, 1)`` rather than negating it (verified against
the registry in ``pmcprg/copulas/_base.py``, not assumed).

Reference: Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed.,
Springer, ch. 2 (survival copulas, Ĉ(u,v) = u + v − 1 + C(1−u, 1−v)).

Numerics
--------
The wrapper used to call the base copula at ``(1.0 − u, 1.0 − v)`` formed
in floating point. For u = 10⁻¹² that is ``1 − 1.000089·10⁻¹²``: the
complement the base needs (log u for Joe, −log(1 − u) for GH and Clayton)
loses four digits, which put the log-densities of SGH and SJoe off by up
to 4·10⁻⁴ and 8·10⁻⁴ nat and SGH's h-function by 2.5·10⁻⁸ on the grid
of ``test_copula_limits.py`` (audit FR-1, roadmap technique 8). The base families now expose a *kernel interface* — the
complement is never formed, the kernel coordinate of 1 − u is computed from
u itself:

``_kcoord(x)`` / ``_kcoord_reflected(x)``
    kernel coordinate of x, and of 1 − x computed from x
    (``log x`` and ``log1p(−x)`` for Clayton and GH; the reverse for Joe);
``_k_logpdf(ka, kb)``
    log c(a, b);
``_k_cdf(ka, kb)``, ``_k_h(kb, ka)``
    ``(C, 1 − C)`` and ``(h(b|a), 1 − h(b|a))``, each member to relative
    precision;
``_k_inv_h(lw, ka)``
    ``(b, 1 − b)`` solving h(b | a) = w, from ``lw = log w``.

so that ĉ(u, v) = c from ``_kcoord_reflected(u), _kcoord_reflected(v)``,
ĥ(v|u) is the ``1 − h`` member, inv ĥ is the ``1 − b`` member of the base
inverse at ``lw = log1p(−w)``, and

    Ĉ(u, v) = (u − (1 − v)) + C     if C ≤ 1/2,
              (u + v) − (1 − C)     otherwise,

whose rounding error is ε·max(|u − (1 − v)|, C) or ε·max(u + v, 1 − C) —
proportional to Ĉ in both corners where it is small, instead of ε.
"""
if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import logging
import numpy as np

from pmcprg.copulas._base                   import CopulaVirt, FitResult
from pmcprg.copulas.archimedean.bb1         import CopulaBB1
from pmcprg.copulas.archimedean.clayton     import CopulaClayton
from pmcprg.copulas.archimedean.gumbel      import CopulaGH
from pmcprg.copulas.archimedean.joe         import CopulaJoe
from pmcprg.numerics                     import EPS, ONE_MINUS_EPS, minmaxEPS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generic wrapper
# ---------------------------------------------------------------------------

class SurvivalCopula(CopulaVirt):
    """180°-rotation wrapper around a base copula exposing the kernel interface.

    Subclasses must set the class attribute ``_base_class`` to a family that
    implements ``_kcoord_reflected``, ``_k_logpdf``, ``_k_cdf``, ``_k_h`` and
    ``_k_inv_h`` (module docstring).
    """

    _base_class: type[CopulaVirt] | None = None

    _KERNEL_INTERFACE = ('_kcoord_reflected', '_k_logpdf', '_k_cdf', '_k_h', '_k_inv_h')

    def __init__(self, **kwargs):
        missing = [m for m in self._KERNEL_INTERFACE if not hasattr(self._base_class, m)]
        if missing:
            raise TypeError(f"{type(self).__name__}: base {getattr(self._base_class, '__name__', None)} "
                            f"lacks the kernel interface {missing} (see pmcprg.copulas.archimedean.survival).")
        # Build base copula first; _update_params (called by super) will sync it.
        self._base: CopulaVirt = self._base_class(**kwargs)  # type: ignore[misc]
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    def _update_params(self):
        """Propagate current params to the base copula and mirror theta."""
        self._base.params = self.params
        self._base._update_params()
        if hasattr(self._base, 'theta'):
            self.theta = self._base.theta
        if hasattr(self._base, 'delta'):
            self.delta = self._base.delta

    # ------------------------------------------------------------------
    # Kernel interface (audit FR-8, closing round) — lets a SurvivalCopula
    # itself serve as the ``_base_class`` of a 90°/270° rotation
    # (``pmcprg.copulas.archimedean.rotated.RotatedCopula``), the way
    # ``SurvivalBB1`` does for ``SurvivalBB190``/``SurvivalBB1270``. Own
    # natural kernel coordinate of x: since Ŝ(u,v) = u+v−1+C(1−u,1−v), Ŝ's
    # own coordinate of x is C's coordinate of (1−x) — exactly ``_base``'s
    # own ``_kcoord_reflected``/``_kcoord`` with the two swapped (kS(x) =
    # kC(1−x), kS(1−x) = kC(x)).
    # ------------------------------------------------------------------
    def _kcoord(self, x):
        return self._base._kcoord_reflected(x)

    def _kcoord_reflected(self, x):
        return self._base._kcoord(x)

    def _k_logpdf(self, ka, kb):
        """ĉ(a,b) = c(1−a,1−b): with ka = kS(a) = kC(1−a) (kb likewise),
        this is exactly ``_base``'s own ``_k_logpdf`` at the same (ka, kb) —
        no further transform, same substitution as :meth:`_logpdf`."""
        return self._base._k_logpdf(ka, kb)

    def _k_cdf(self, ka, kb):
        """(Ŝ(a,b), 1 − Ŝ(a,b)) from ka = kC(1−a), kb = kC(1−b): a and b are
        recovered from ka, kb via ``expm1`` (exact — ``exp(ka) = 1 − a``
        already, ``expm1`` only sharpens it near a = 0), then combined with
        the base's C/complement at the same (ka, kb) by the same
        branch-on-C trick as :meth:`_cdf`."""
        a = -np.expm1(ka)
        b = -np.expm1(kb)
        c, cbar = self._base._k_cdf(ka, kb)
        with np.errstate(over='ignore', invalid='ignore'):
            out = np.where(c <= 0.5, (a - (1.0 - b)) + c, (a + b) - cbar)
        out = np.clip(out, 0.0, 1.0)
        return out, 1.0 - out

    def _k_h(self, kb, ka):
        """(ĥ(b|a), 1 − ĥ(b|a)): ĥ(v|u) = 1 − h_C(1−v|1−u) (module
        docstring) — the base's own ``_k_h`` at the same (kb, ka), its two
        members swapped."""
        h, hbar = self._base._k_h(kb, ka)
        return hbar, h

    def _k_inv_h(self, lw, ka):
        """(b, 1 − b) solving ĥ(b|a) = w, from lw = log w: ĥ(v|u) = w ⟺
        h_C(1−v|1−u) = 1 − w (:meth:`inv_h`'s own derivation), so this is
        the base's own ``_k_inv_h`` at ``log(1 − w) = log1p(−w)`` and the
        same ``ka``, its two members swapped."""
        w = np.exp(lw)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            b_val, bbar_val = self._base._k_inv_h(np.log1p(-w), ka)
        return bbar_val, b_val

    # ------------------------------------------------------------------
    # Joint-constraint delegation (audit G1, first needed by SurvivalBB1 —
    # BB1 is the first base family here with more than τ alone). No sign
    # flip is needed, unlike ``RotatedCopula``'s own version of this hook:
    # the 180° rotation leaves τ's sign unchanged (module docstring), so the
    # base's admissible set *is* the survival family's admissible set,
    # params passed through as-is. For Clayton/GH/Joe, whose base
    # ``constrain_params``/``constructible_params`` are :class:`CopulaVirt`'s
    # identity, this changes nothing (the dict comes back unchanged).
    # ------------------------------------------------------------------
    @classmethod
    def constrain_params(cls, params: dict) -> dict:
        return cls._base_class.constrain_params(params)

    @classmethod
    def constructible_params(cls, params: dict) -> dict:
        return cls._base_class.constructible_params(params)

    @staticmethod
    def _columns(uv):
        uv = np.asarray(uv, dtype=float)
        return (np.clip(uv[:, 0], EPS, ONE_MINUS_EPS),
                np.clip(uv[:, 1], EPS, ONE_MINUS_EPS))

    def _logpdf(self, u, v):
        b = self._base
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            return b._k_logpdf(b._kcoord_reflected(u), b._kcoord_reflected(v))

    def _cdf(self, u, v):
        b = self._base
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            c, cbar = b._k_cdf(b._kcoord_reflected(u), b._kcoord_reflected(v))
            out = np.where(c <= 0.5, (u - (1.0 - v)) + c, (u + v) - cbar)
        return np.clip(out, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Core functions — base kernel at the reflected coordinates
    # ------------------------------------------------------------------

    def pdf(self, uv):
        """ĉ(u,v) = c(1−u, 1−v)."""
        u = np.float64(minmaxEPS(uv[0]))
        v = np.float64(minmaxEPS(uv[1]))
        with np.errstate(over='ignore'):
            return float(np.exp(self._logpdf(u, v)))

    def pdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised ĉ(u,v) = c(1−u, 1−v), ``exp`` of :meth:`logpdf_array` (no floor)."""
        return np.exp(self.logpdf_array(uv))

    def logpdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised log ĉ(u,v) = log c(1−u, 1−v), complements never formed."""
        return self._logpdf(*self._columns(uv))

    def cdf(self, uv):
        """Ĉ(u,v) = u + v − 1 + C(1−u, 1−v)."""
        u = np.float64(minmaxEPS(uv[0]))
        v = np.float64(minmaxEPS(uv[1]))
        return float(self._cdf(u, v))

    def cdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised Ĉ(u,v) = u + v − 1 + C(1−u, 1−v) (module docstring)."""
        return self._cdf(*self._columns(uv))

    def conditional_cdf(self, v: float, u: float) -> float:
        """ĥ(v|u) = 1 − h_C(1−v | 1−u), the ``1 − h`` member of the base kernel."""
        b = self._base
        u = np.float64(minmaxEPS(u))
        v = np.float64(minmaxEPS(v))
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            _, hbar = b._k_h(b._kcoord_reflected(v), b._kcoord_reflected(u))
        return float(np.clip(hbar, 0.0, 1.0))

    def inv_h(self, w: float, u: float) -> float:
        """Inverse h-function via the survival relation.

        Solving ĥ(v|u) = w means ``1 − h_C(1−v | 1−u) = w``, i.e.
        ``h_C(1−v | 1−u) = 1−w``, so ``v = 1 − inv_h_base(1 − w, 1 − u)`` —
        taken as the ``1 − b`` member of the base kernel inverse, evaluated at
        ``log(1 − w) = log1p(−w)`` and the reflected coordinate of u.
        """
        return float(self.inv_h_array(np.array([w], dtype=float), np.array([u], dtype=float))[0])

    def inv_h_array(self, w: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Vectorised version: v = 1 − inv_h_base(1 − w, 1 − u), complements never formed."""
        b = self._base
        w = np.clip(np.asarray(w, dtype=float), EPS, ONE_MINUS_EPS)
        u = np.clip(np.asarray(u, dtype=float), EPS, ONE_MINUS_EPS)
        if w.shape != u.shape:
            raise ValueError(f"inv_h_array: w {w.shape} and u {u.shape} must match.")
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            _, vbar = b._k_inv_h(np.log1p(-w), b._kcoord_reflected(u))
        return np.clip(vbar, EPS, ONE_MINUS_EPS)

    def tail_dependence(self) -> tuple[float, float]:
        """Swap λ_L and λ_U of the base copula."""
        lam_L, lam_U = self._base.tail_dependence()
        return lam_U, lam_L


# ---------------------------------------------------------------------------
# Concrete survival copulas
# ---------------------------------------------------------------------------

class SurvivalClayton(SurvivalCopula):
    """Survival Clayton copula — upper-tail dependence.

    Base: Clayton (lower-tail) → rotation gives upper-tail.
    λ_L = 0, λ_U = 2^{−1/θ_Clayton}.
    """
    _base_class = CopulaClayton


class SurvivalGH(SurvivalCopula):
    """Survival Gumbel-Hougaard copula — lower-tail dependence.

    Base: GH (upper-tail) → rotation gives lower-tail.
    λ_L = 2 − 2^{1/θ_GH}, λ_U = 0.
    """
    _base_class = CopulaGH


class SurvivalJoe(SurvivalCopula):
    """Survival Joe copula — lower-tail dependence.

    Base: Joe (upper-tail) → rotation gives lower-tail.
    λ_L = 2 − 2^{1/θ_Joe}, λ_U = 0.
    """
    _base_class = CopulaJoe


class SurvivalBB1(SurvivalCopula):
    """Survival BB1 copula (FR-8, last round) — tail roles swapped, still
    two-sided.

    Base: BB1 (λ_L Clayton-like, λ_U Gumbel-like) → rotation swaps the two:
    λ_L(Ŝ) = λ_U(BB1) = 2 − 2^{1/δ}, λ_U(Ŝ) = λ_L(BB1) = 2^{−1/(θδ)}
    (:meth:`SurvivalCopula.tail_dependence`). BB1 is exchangeable
    (``c(u,v) = c(v,u)``, module docstring of ``bb1.py``), and survival
    preserves exchangeability (the reflection ``(u,v) → (1−u,1−v)`` is itself
    symmetric in ``u``/``v``): ``SurvivalBB1`` is exchangeable too — unlike
    its own 90°/270° rotations below, which are not (module docstring of
    ``rotated.py``).

    ``delta`` (δ ≥ 1) passes through :meth:`SurvivalCopula._update_params`
    unchanged, same as every other parameter; τ is unchanged by the 180°
    rotation (module docstring), so ``TAU_MIN_MAX`` mirrors ``CopulaBB1``'s
    own ``[0 + ε, 1)`` bit for bit, and ``constrain_params``/
    ``constructible_params`` delegate to ``CopulaBB1``'s joint ``δ <
    1/(1 − τ)`` hook unchanged (class docstring above).
    """
    _base_class = CopulaBB1
    n_params: int = 2

    @classmethod
    def fit(cls, data: np.ndarray, method: str = 'mle', *, weights=None,
            pseudo_obs: bool = False) -> 'FitResult':
        """Delegate to :meth:`CopulaBB1.fit` on the ``(1 − u, 1 − v)``-
        reflected data — module docstring: ``(U,V) ~ SurvivalBB1`` iff
        ``(1−U,1−V) ~ BB1``. τ is *not* negated back (unchanged by a 180°
        rotation, unlike the 90°/270° case's ``CopulaBB190``/
        ``CopulaBB1270.fit``, which this mirrors otherwise).

        Needed, at the time, for the same reason those two override ``fit``:
        BB1's own ``fit`` always used MLE regardless of ``method``, while the
        generic ``CopulaVirt.fit(method='tau')`` built the final copula at the
        *default* δ = 1.5 without projecting it through
        :meth:`constructible_params` (δ = 1.5 is inadmissible at any
        τ ≤ 1/3). Since FR-12 ``method='tau'`` (itau: δ by maximum likelihood
        at τ̂, on its admissible interval) and the weighted fits take the
        generic path; the unweighted MLE still reuses BB1's validated fit via
        the reflection identity, as the 90°/270° rotations do.
        """
        if weights is not None or method == 'tau':
            # itau and the weighted fits: the generic path (FR-12).
            return super().fit(data, method=method, weights=weights, pseudo_obs=pseudo_obs)
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError(f'data must be shape (n, 2), got {data.shape}.')
        reflected = 1.0 - data
        base_fit = CopulaBB1.fit(reflected, method=method, pseudo_obs=pseudo_obs)
        tau_k = base_fit.tau_k
        cop = cls(tau_k=tau_k, delta=base_fit.copula.delta)
        uv = 1.0 - base_fit.uv
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            log_lik = float(np.sum(cop.logpdf_array(uv)))
        return FitResult(copula=cop, method='mle', tau_k=tau_k,
                         log_likelihood=log_lik, n_obs=base_fit.n_obs, uv=uv,
                         converged=base_fit.converged, message=base_fit.message,
                         n_iter=base_fit.n_iter, n_eval=base_fit.n_eval)


# ---------------------------------------------------------------------------
# Quick smoke test / demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from pathlib import Path

    for cls, name in [
        (SurvivalClayton, 'SurvivalClayton'),
        (SurvivalGH,      'SurvivalGH'),
        (SurvivalJoe,     'SurvivalJoe'),
        (SurvivalBB1,     'SurvivalBB1'),
    ]:
        kwargs = {'tau_k': 0.5, 'delta': 1.5} if cls is SurvivalBB1 else {'tau_k': 0.5}
        cop = cls(**kwargs)
        lam_L, lam_U = cop.tail_dependence()
        print(f'\n--- {name} ---')
        print(f'tau_k    : {cop.params["tau_k"]:.4f}  range={cop.tau_range}')
        print(f'theta    : {cop.theta:.6f}')
        print(f'pdf(0.3, 0.7) = {cop.pdf([0.3, 0.7]):.6f}')
        print(f'cdf(0.3, 0.7) = {cop.cdf([0.3, 0.7]):.6f}')
        print(f'h(0.7 | 0.3)  = {cop.conditional_cdf(0.7, 0.3):.6f}')
        print(f'tail dep : λ_L = {lam_L:.4f},  λ_U = {lam_U:.4f}')

        plot_dir = Path('./data/Plots/Copulas')
        plot_dir.mkdir(parents=True, exist_ok=True)
        cop.plot_pdf(plot_dir)
        cop.plot_cdf(plot_dir)
        cop.plot_overview(plot_dir)
