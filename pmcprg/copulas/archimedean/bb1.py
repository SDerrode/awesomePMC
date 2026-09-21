"""
BB1 copula (Clayton-Gumbel) — lower- AND upper-tail dependence.

Generator: φ(t) = (t^{-θ} − 1)^δ,   θ > 0, δ ≥ 1.
CDF: C(u,v) = (1 + ((u^{-θ}−1)^δ + (v^{-θ}−1)^δ)^{1/δ})^{−1/θ}

Closed-form relationships:
    τ_K = 1 − 2/(δ(θ+2))   →   θ = 2/(δ(1−τ_K)) − 2
    λ_L = 2^{−1/(θδ)}          (lower tail, Clayton-like)
    λ_U = 2 − 2^{1/δ}          (upper tail, Gumbel-like)

Special cases:
    δ = 1 → Clayton (λ_U = 0)
    θ → 0 with δ = 1 → independence
    θ → 0 (then δ = 1/(1−τ_K)) → Gumbel–Hougaard: λ_L → 0, λ_U = 2 − 2^{1/δ}
    δ → ∞ at fixed θ → comonotonicity: λ_L → 1 and λ_U → 1
At fixed τ_K, θδ = 2/(1−τ_K) − 2δ, so a larger δ moves dependence from the
lower tail (λ_L falls) to the upper tail (λ_U rises). The former line
"δ → ∞ → Gumbel-like (λ_L → 0)" had the limits crossed: with
λ_L = 2^{−1/(θδ)}, δ → ∞ sends λ_L to 1, and the Gumbel limit is θ → 0
(audit K-10).

Parameters stored in `params`:
    tau_k : Kendall's τ ∈ (1 − 1/δ, 1)   [ensures θ > 0]
    delta : δ ≥ 1 (default 1.5 if omitted)

n_params = 2 — ``fit`` is the 2-D MLE over (θ, δ) by default; ``method='tau'``
is itau, τ̂ then δ by maximum likelihood at that τ (FR-12).

Reference: Joe, H. (1997). *Multivariate Models and Dependence Concepts*,
Chapman & Hall, ch. 5 (family BB1: Clayton at δ = 1, Gumbel as θ → 0) and
ch. 2 (tail dependence).

Numerics
--------
With A = u^{−θ} − 1, B = v^{−θ} − 1, P = A^δ + B^δ, S = P^{1/δ}, the textbook
expressions (Joe 1997, ch. 5)

    C(u,v)  = (1 + S)^{−1/θ},
    c(u,v)  = (uv)^{−θ−1} (AB)^{δ−1} S^{1−2δ} (1+S)^{−1/θ−2} [θ(δ−1) + (θδ+1) S],
    h(v|u)  = u^{−θ−1} A^{δ−1} S^{1−δ} (1+S)^{−1/θ−1},

fail at both ends when formed in linear scale: ``u^{−θ} − 1`` cancels as
u → 1 (four digits left at u = 1 − 10⁻¹², log c off by 10⁻⁴ nat), and the
powers overflow as u → 0 — the scalar pdf returned its ``EPS`` sentinel at
(10⁻¹², 10⁻¹²) for τ = 0.9 where c = 4.3·10¹², and ``logpdf_array`` returned
NaN on 17/20 and 20/20 points at τ = 0.999 and 0.9999 (audit RB-8). No power
is formed here: with a = −θ log u > 0,

    log A = a + log(−expm1(−a))                 (exact as a → 0, no overflow as a → ∞),
    log S = logaddexp(δ log A, δ log B) / δ,    log(1 + S) = logaddexp(0, log S),

the log-space evaluation of Archimedean densities of Hofert, Mächler &
McNeil (2012), *J. Multivariate Anal.* 110, 133–150,
doi:10.1016/j.jmva.2012.02.019. ``C(u, 1) = u`` holds to rounding: B → 0
gives log B → −∞ and logaddexp returns δ log A (audit K-12). There is no
fallback: a value that cannot be computed is NaN, not ``EPS`` or a 0/1 step
(audit RB-9).

Kernel interface (audit FR-8, BB1 round). Every path above is built from
``ka = log u``, ``kb = log v`` — never from ``u``/``v`` themselves — because
``A = u^{-θ} − 1`` is already a function of ``log u`` alone
(``_bb1_loga``); this is BB1's *own* natural kernel coordinate, and it
coincides with Gumbel's (``pmcprg.copulas.archimedean.gumbel``: ``ka =
log u``), not with Joe's (``ka = log(1 − u)``), because BB1's generator
``φ(t) = (t^{-θ} − 1)^δ`` — unlike Joe's ``φ(t) = -log(1-(1-t)^θ)`` — is
built from ``t`` directly, not from its complement. ``_kcoord_reflected(x)
= log1p(-x)`` is then the exact log of ``1 − x`` for the 90°/270° rotation
wrapper (``pmcprg.copulas.archimedean.rotated``) to pass as ``ka``/``kb``
without ever forming ``1 − x`` in linear scale.

The two-parameter generator changes only which quantity the *inverse*
h-function is solved for, not the coordinate: ``h(v|u) = w`` has no closed
form in BB1 (unlike Gumbel/Joe, whose textbook h admits an explicit inverse
up to a monotone Newton iteration in a transformed variable — Hofert et al.
2012's substitution does not separate the two parameters θ, δ enough for
that here). ``_k_inv_h`` therefore solves ``log h(v|u) = log w`` by Brent's
method directly on ``kb = log v`` — the same kernel coordinate the rest of
the interface uses, so a reflected conditioning value never re-forms ``1 −
u`` either — rather than by a closed-form step. This is a genuine, exact
(to Brent's tolerance) solve, not an approximation: BB1's own public
``inv_h``/``inv_h_array`` (unchanged by this refactor) already used the
generic numerical inversion of ``CopulaVirt.inv_h`` via ``conditional_cdf``,
and ``_k_inv_h`` is the same solve reformulated on kernel coordinates so
``CopulaBB190``/``CopulaBB1270`` (``rotated.py``) can use it. Both stop Brent
at ``xtol = 1e-15`` (FR-11; the kernel inverse had 1e-13, the generic one
1e-8): against mpmath, 7.1e-15 for the kernel inverse and 6.4e-15 for the
generic one on the parity grid, both at the rounding floor of h where it is
flat.
"""
if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import logging
import math

import numpy as np
from scipy.optimize import brentq, minimize

from pmcprg.copulas._base import TAU_PAD_ABS, TAU_PAD_REL, CopulaVirt, FitResult
from pmcprg.exceptions    import CopulaParameterError
from pmcprg.numerics   import EPS, ONE_MINUS_EPS, minmaxEPS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Log-space kernel on ka = log u, kb = log v (module docstring) — shared by
# every evaluation path, including the rotation wrapper's kernel interface.
# ---------------------------------------------------------------------------

def _bb1_loga(ka, th):
    """``log(u^{−θ} − 1)`` from ``ka = log u``: a + log(1 − e^{−a}), a = −θ·ka."""
    a = -th * ka
    return a + np.log(-np.expm1(-a))


def _bb1_terms_k(ka, kb, th, de):
    """``(ka, log A, log B, log S, log(1 + S))`` from ``ka = log u``, ``kb = log v``."""
    log_a = _bb1_loga(ka, th)
    log_b = _bb1_loga(kb, th)
    log_s = np.logaddexp(de * log_a, de * log_b) / de
    return ka, log_a, log_b, log_s, np.logaddexp(0.0, log_s)


def _bb1_logpdf_k(ka, kb, th, de):
    """log c(u, v) from ``ka = log u``, ``kb = log v``."""
    ka, log_a, log_b, log_s, log_1ps = _bb1_terms_k(ka, kb, th, de)
    with np.errstate(divide='ignore'):
        # θ(δ − 1) + (θδ + 1) S: at δ = 1 (Clayton) the constant vanishes, and
        # logaddexp(−inf, x) = x handles that boundary exactly.
        log_last = np.logaddexp(np.log(th * (de - 1.0)), np.log(th * de + 1.0) + log_s)
    return ((-th - 1.0) * (ka + kb)
            + (de - 1.0) * (log_a + log_b)
            + (1.0 - 2.0 * de) * log_s
            + (-1.0 / th - 2.0) * log_1ps
            + log_last)


def _bb1_cdf_k(ka, kb, th, de):
    """``(C, 1 − C)``, ``C = (1 + S)^{−1/θ} = exp(−log(1 + S)/θ)``."""
    e = _bb1_terms_k(ka, kb, th, de)[4] / th
    return np.exp(-e), -np.expm1(-e)


def _bb1_logh_k(kb, ka, th, de):
    """log h(v|u) = (−θ−1)·ka + (δ−1)·log A + (1−δ)·log S + (−1/θ−1)·log(1+S),
    from ``kb = log v``, ``ka = log u`` (GH/Joe argument order: conditioned
    variable first, conditioning variable second)."""
    _, log_a, _, log_s, log_1ps = _bb1_terms_k(ka, kb, th, de)
    return ((-th - 1.0) * ka
            + (de - 1.0) * log_a
            + (1.0 - de) * log_s
            + (-1.0 / th - 1.0) * log_1ps)


def _bb1_h_k(kb, ka, th, de):
    """``(h, 1 − h)`` of h(v|u), in log space (module docstring)."""
    logh = _bb1_logh_k(kb, ka, th, de)
    return np.exp(logh), -np.expm1(logh)


def _bb1_inv_h_k(lw, ka, th, de):
    """``(v, 1 − v)`` solving h(v|u) = w, from ``lw = log w`` and ``ka = log u``.

    No closed form (module docstring): Brent's method on ``kb = log v`` — the
    same kernel coordinate as everything else here — bracketed on
    ``[log EPS, log(1 − EPS)]``, the full admissible range of ``v``. ``log
    h(v|u)`` is monotone increasing in ``kb`` (h is a CDF in v, v = exp(kb)
    increasing in kb), so the bracket always contains the root when one
    exists; ``w`` outside the attainable range saturates to the nearer end,
    matching :meth:`CopulaVirt.inv_h`'s own saturation convention.
    """
    lw = np.asarray(lw, dtype=float)
    ka = np.asarray(ka, dtype=float)
    shape = np.broadcast(lw, ka).shape
    lw_flat = np.broadcast_to(lw, shape).ravel()
    ka_flat = np.broadcast_to(ka, shape).ravel()
    kb_lo, kb_hi = math.log(EPS), math.log(ONE_MINUS_EPS)
    out = np.empty(lw_flat.shape, dtype=float)
    for i in range(lw_flat.size):
        l_, k_ = float(lw_flat[i]), float(ka_flat[i])
        logh_lo = _bb1_logh_k(kb_lo, k_, th, de)
        logh_hi = _bb1_logh_k(kb_hi, k_, th, de)
        if l_ <= logh_lo:
            out[i] = kb_lo
        elif l_ >= logh_hi:
            out[i] = kb_hi
        else:
            out[i] = brentq(lambda kb_: _bb1_logh_k(kb_, k_, th, de) - l_,
                             kb_lo, kb_hi, maxiter=100, xtol=1e-15,
                             rtol=8.0 * np.finfo(float).eps)
    kb = out.reshape(shape)
    return np.exp(kb), -np.expm1(kb)


class CopulaBB1(CopulaVirt):
    """BB1 (Clayton-Gumbel) copula — lower- AND upper-tail dependence.

    Parameters
    ----------
    tau_k : float
        Kendall's τ.  Must satisfy τ > 1 − 1/δ so that θ > 0.
    delta : float, optional
        Shape parameter δ ≥ 1 (default 1.5 if omitted).
        δ = 1 reduces to Clayton; at fixed τ a larger δ shifts dependence
        from the lower to the upper tail (see the module docstring).
    """

    n_params: int = 2

    # Smallest θ treated as admissible; θ = 0 is the degenerate boundary.
    _THETA_FLOOR: float = 1e-6

    def __init__(self, **kwargs):
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    @classmethod
    def delta_max(cls, tau: float) -> float:
        """Largest δ admissible at this τ, from ``θ = 2/(δ(1−τ)) − 2 > 0``.

        ``θ ≥ _THETA_FLOOR`` rearranges to ``δ ≤ 2/((θ_floor+2)(1−τ))``, which
        tends to ``1/(1−τ)``. As τ → 0 it tends to 1, pinning δ to the Clayton
        limit — which is exactly what BB1 degenerates to there.
        """
        gap = max(1.0 - float(tau), EPS)
        return 2.0 / ((cls._THETA_FLOOR + 2.0) * gap)

    @classmethod
    def constrain_params(cls, params: dict) -> dict:
        """Clamp δ into ``[1, delta_max(τ)]`` — see :meth:`delta_max`.

        δ and τ are *jointly* constrained, which a box-bounded optimiser
        cannot express: the default δ = 1.5 is inadmissible for any τ ≤ 1/3,
        so a fit on moderately-dependent data used to start on the failure
        plateau and never move (δ came back as the untouched initial value).
        """
        out = dict(params)
        tau = float(out.get("tau_k", 0.0))
        delta = float(out.get("delta", 1.5))
        hi = max(1.0, cls.delta_max(tau))
        out["delta"] = float(np.clip(delta, 1.0, hi))
        return out

    @classmethod
    def constructible_params(cls, params: dict) -> dict:
        """``params`` itself when BB1 builds at it; else δ moved inside ``[1, 1/(1 − τ))``.

        The test is the constructor's own (:meth:`_update_params`, same
        floating-point operations): ``δ ≥ 1`` and ``θ = 2/(δ(1 − τ)) − 2 > 0``.
        Only a refused pair changes, and only δ — τ is the value the caller
        drew (multistart jitter, family draw) and stays. The admissible
        δ-interval at τ is ``[1, 1/(1 − τ))``: δ = 1 (Clayton) is a member,
        ``1/(1 − τ)`` (θ = 0, the Gumbel limit) is refused. δ goes to that
        interval with its refused end pulled in exactly as
        :func:`pmcprg.copulas._base.constructible_tau_range` pulls in a
        refused τ end: by ``max(TAU_PAD_REL · span, TAU_PAD_ABS)`` of the
        interval's span ``τ/(1 − τ)``. The repaired δ is off the boundary by
        the fraction ``TAU_PAD_REL`` of the interval whatever its scale
        (θ ≈ 2·10⁻⁴·τ; the absolute floor takes over below τ ≈ 10⁻⁴), not on
        it; below τ ≈ 10⁻⁸ the interval is narrower than the pad and δ = 1.
        A missing δ is the constructor's default 1.5. A pair no δ repairs
        (τ ≥ 1) is returned unchanged, for the constructor to refuse.
        """
        tau = float(params["tau_k"])
        delta = float(params.get("delta", 1.5))
        denom = delta * (1.0 - tau)
        if not (delta < 1.0 or denom <= 0.0 or 2.0 / denom - 2.0 <= 0.0):
            return params
        if not tau < 1.0:
            return params
        hi = 1.0 / (1.0 - tau)
        pad = max(TAU_PAD_REL * (hi - 1.0), TAU_PAD_ABS)
        return {**params, "delta": max(1.0, min(delta, hi - pad))}

    def _update_params(self):
        tau   = self.params['tau_k']
        delta = float(self.params.get('delta', 1.5))
        if delta < 1.0:
            raise CopulaParameterError(f'BB1: delta must be ≥ 1, got {delta:.4f}')
        denom = delta * (1.0 - tau)
        if denom <= 0.0:
            raise CopulaParameterError(
                f'BB1: delta*(1−tau) = {denom:.6f} ≤ 0.'
            )
        theta = 2.0 / denom - 2.0
        if theta <= 0.0:
            raise CopulaParameterError(
                f'BB1: theta = {theta:.4f} ≤ 0 for tau={tau:.4f}, delta={delta:.4f}. '
                f'Need tau > 1 − 1/delta = {1.0 - 1.0/delta:.4f}.'
            )
        self.theta = theta
        self.delta = delta
        # Store delta back so it's preserved in self.params for plot_multi_tau etc.
        self.params['delta'] = delta

    # ------------------------------------------------------------------
    # Kernel interface (audit FR-8) — consumed by the 90°/270° rotation
    # wrapper in ``pmcprg.copulas.archimedean.rotated``, same shape as
    # ``CopulaGH``/``CopulaJoe``'s (module docstring: BB1's own natural
    # kernel coordinate is ``ka = log u``, coinciding with Gumbel's).
    # ------------------------------------------------------------------

    @staticmethod
    def _kcoord(x):
        return np.log(x)

    @staticmethod
    def _kcoord_reflected(x):
        return np.log1p(-x)

    def _k_logpdf(self, ka, kb):
        return _bb1_logpdf_k(ka, kb, self.theta, self.delta)

    def _k_cdf(self, ka, kb):
        return _bb1_cdf_k(ka, kb, self.theta, self.delta)

    def _k_h(self, kb, ka):
        return _bb1_h_k(kb, ka, self.theta, self.delta)

    def _k_inv_h(self, lw, ka):
        return _bb1_inv_h_k(lw, ka, self.theta, self.delta)

    # ------------------------------------------------------------------
    # CDF / PDF / h-function — all built from the kernel interface above
    # ------------------------------------------------------------------

    def cdf(self, uv):
        """C(u,v) = (1 + S)^{−1/θ}."""
        u = minmaxEPS(uv[0])
        v = minmaxEPS(uv[1])
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            c, _ = self._k_cdf(self._kcoord(np.float64(u)), self._kcoord(np.float64(v)))
        return float(np.clip(c, 0.0, 1.0))

    def cdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised closed-form CDF (BB1 has no statsmodels backend)."""
        uv = np.asarray(uv, dtype=float)
        u  = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v  = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            c, _ = self._k_cdf(self._kcoord(u), self._kcoord(v))
        return np.clip(c, 0.0, 1.0)

    def pdf(self, uv):
        """c(u,v) = u^{-θ-1}·v^{-θ-1}·A^{δ-1}·B^{δ-1}·S^{1-2δ}·(1+S)^{-1/θ-2}·[θ(δ-1)+(θδ+1)S].

        Evaluated as ``exp`` of the log-space kernel; 0.0 only on underflow.
        """
        u = minmaxEPS(uv[0])
        v = minmaxEPS(uv[1])
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            return float(np.exp(self._k_logpdf(self._kcoord(np.float64(u)),
                                                self._kcoord(np.float64(v)))))

    def pdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised closed-form PDF, ``exp`` of :meth:`logpdf_array` (no floor)."""
        return np.exp(self.logpdf_array(uv))

    def logpdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Native log-PDF, computed entirely in log space (see module docstring).

        With A = u^{−θ} − 1, B = v^{−θ} − 1, S = (A^δ + B^δ)^{1/δ}:
            log c(u, v) = (−θ − 1)(log u + log v)
                        + (δ − 1)(log A + log B)
                        + (1 − 2δ) log S
                        + (−1/θ − 2) log(1 + S)
                        + log(θ(δ − 1) + (θδ + 1) S)
        """
        uv = np.asarray(uv, dtype=float)
        u  = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v  = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            return self._k_logpdf(self._kcoord(u), self._kcoord(v))

    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = u^{-θ-1} · A^{δ-1} · S^{1-δ} · (1+S)^{-1/θ-1}."""
        u = minmaxEPS(u)
        v = minmaxEPS(v)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            h, _ = self._k_h(self._kcoord(np.float64(v)), self._kcoord(np.float64(u)))
        return float(np.clip(h, 0.0, 1.0))

    def tail_dependence(self) -> tuple[float, float]:
        """λ_L = 2^{−1/(θδ)},  λ_U = 2 − 2^{1/δ}."""
        return (
            float(2.0 ** (-1.0 / (self.theta * self.delta))),
            float(2.0 - 2.0 ** (1.0 / self.delta)),
        )

    # ------------------------------------------------------------------
    # 2-D MLE fitting (overrides CopulaVirt.fit)
    # ------------------------------------------------------------------

    @classmethod
    def fit(cls, data: np.ndarray, method: str = 'mle', *, weights=None,
            pseudo_obs: bool = False) -> 'FitResult':
        """Fit BB1 by 2-D MLE over (θ, δ), or by itau.

        ``method='mle'`` (default, unweighted): the 2-D MLE below. The
        log-likelihood is ``Σ logpdf_array`` — no ``EPS`` floor on the
        density; an inadmissible (θ, δ) or a non-finite sum gets the finite
        penalty the bounded optimiser needs.

        ``method='tau'``: τ̂ from Kendall's τ, then δ by maximum likelihood at
        that τ (a profile likelihood, FR-12; before, ``'tau'`` warned and ran
        this MLE). ``weights`` and ``pseudo_obs`` as in :meth:`CopulaVirt.fit`.

        Returns
        -------
        FitResult
        """
        from scipy.stats import kendalltau as _kendalltau

        from pmcprg.copulas._base import _pseudo_observations

        if weights is not None or method == 'tau':
            return super().fit(data, method=method, weights=weights, pseudo_obs=pseudo_obs)
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError(f'data must be shape (n, 2), got {data.shape}.')
        n = data.shape[0]
        if n < 8:
            raise ValueError(
                f'BB1 requires at least 8 observations for 2-D MLE, got {n}.'
            )
        if method != 'mle':
            raise ValueError(f"method must be 'tau' or 'mle', got {method!r}.")

        uv = _pseudo_observations(data, pseudo_obs)

        # Initial point: empirical τ → choose δ₀=1.5, θ₀ from formula
        tau_emp, _ = _kendalltau(data[:, 0], data[:, 1])
        tau_emp    = float(np.clip(tau_emp, 0.05, 0.95))
        delta0     = 1.5
        theta0     = max(1e-3, 2.0 / (delta0 * (1.0 - tau_emp)) - 2.0)

        def _neg_ll(x: np.ndarray) -> float:
            th, de = float(x[0]), float(x[1])
            if th <= 0.0 or de < 1.0:
                return 1e15
            tau_t = float(np.clip(1.0 - 2.0 / (de * (th + 2.0)), EPS, ONE_MINUS_EPS))
            try:
                cop = cls(tau_k=tau_t, delta=de)
            except CopulaParameterError:
                return 1e15
            ll = float(np.sum(cop.logpdf_array(uv)))
            return -ll if np.isfinite(ll) else 1e15

        res = minimize(
            _neg_ll,
            x0=[theta0, delta0],
            method='L-BFGS-B',
            bounds=[(1e-6, None), (1.0, None)],
            options={'ftol': 1e-10, 'gtol': 1e-8, 'maxiter': 400},
        )

        theta_hat = float(max(1e-6, res.x[0]))
        delta_hat = float(max(1.0,  res.x[1]))
        tau_k_hat = float(np.clip(
            1.0 - 2.0 / (delta_hat * (theta_hat + 2.0)),
            EPS, ONE_MINUS_EPS,
        ))

        cop = cls(tau_k=tau_k_hat, delta=delta_hat)
        log_lik = float(np.sum(cop.logpdf_array(uv)))

        return FitResult(
            copula=cop,
            method='mle',
            tau_k=tau_k_hat,
            log_likelihood=log_lik,
            n_obs=n,
            uv=uv,
            converged=bool(res.success),
            message=f"L-BFGS-B over (θ, δ): {res.message}",
            n_iter=int(res.nit),
            n_eval=int(res.nfev),
        )


if __name__ == '__main__':
    from pathlib import Path

    cop = CopulaBB1(tau_k=0.5, delta=1.5)
    lL, lU = cop.tail_dependence()
    print(f'Copula : {cop.copula_enum.value.LONG_NAME}')
    print(f'tau_k  : {cop.params["tau_k"]:.4f}  delta={cop.delta:.4f}  theta={cop.theta:.6f}')
    print(f'tau    check: 1-2/(δ(θ+2)) = {1.0-2.0/(cop.delta*(cop.theta+2.0)):.6f}  (want {cop.params["tau_k"]:.6f})')
    print(f'pdf(0.3, 0.7) = {cop.pdf([0.3, 0.7]):.6f}')
    print(f'cdf(0.3, 0.7) = {cop.cdf([0.3, 0.7]):.6f}')
    print(f'h(0.7 | 0.3)  = {cop.conditional_cdf(0.7, 0.3):.6f}')
    print(f'tail dep : λ_L = {lL:.4f}  [= 2^(-1/(θδ))],  λ_U = {lU:.4f}  [= 2-2^(1/δ)]')

    # δ=1 should reduce to Clayton
    cop_c = CopulaBB1(tau_k=0.5, delta=1.0)
    from pmcprg.copulas.archimedean.clayton import CopulaClayton
    cop_clay = CopulaClayton(tau_k=0.5)
    print('\nδ=1 vs Clayton @ (0.3, 0.7):')
    print(f'  BB1.pdf={cop_c.pdf([0.3,0.7]):.6f}  Clayton.pdf={cop_clay.pdf([0.3,0.7]):.6f}')
    print(f'  BB1.cdf={cop_c.cdf([0.3,0.7]):.6f}  Clayton.cdf={cop_clay.cdf([0.3,0.7]):.6f}')
    lL_c, lU_c = cop_c.tail_dependence()
    lL_clay, lU_clay = cop_clay.tail_dependence()
    print(f'  BB1 tail=({lL_c:.4f},{lU_c:.4f})  Clayton tail=({lL_clay:.4f},{lU_clay:.4f})')

    plot_dir = Path('./data/Plots/Copulas')
    plot_dir.mkdir(parents=True, exist_ok=True)
    cop.plot_pdf(plot_dir)
    cop.plot_cdf(plot_dir)
    cop.plot_h_function(plot_dir)
    cop.plot_samples(plot_dir)
    cop.plot_overview(plot_dir)
    cop.plot_multi_tau(plot_dir)
