"""
BB1 copula (Joe-Clayton) — lower- AND upper-tail dependence.

Generator: φ(t) = (t^{-θ} − 1)^δ,   θ > 0, δ ≥ 1.
CDF: C(u,v) = (1 + ((u^{-θ}−1)^δ + (v^{-θ}−1)^δ)^{1/δ})^{−1/θ}

Closed-form relationships:
    τ_K = 1 − 2/(δ(θ+2))   →   θ = 2/(δ(1−τ_K)) − 2
    λ_L = 2^{−1/(θδ)}          (lower tail, Clayton-like)
    λ_U = 2 − 2^{1/δ}          (upper tail, Joe-like)

Special cases:
    δ = 1 → Clayton (λ_U = 0)
    θ → 0 with δ = 1 → independence
    δ → ∞ → Gumbel-like (λ_L → 0)

Parameters stored in `params`:
    tau_k : Kendall's τ ∈ (1 − 1/δ, 1)   [ensures θ > 0]
    delta : δ ≥ 1 (default 1.5 if omitted)

n_params = 2 — fitting always uses 2-D MLE over (θ, δ).
"""
if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import logging
import numpy as np
from scipy.optimize import minimize

from prg.copulas._base import CopulaVirt, FitResult
from prg.exceptions    import CopulaParameterError
from prg.numerics   import EPS, ONE_MINUS_EPS, minmaxEPS

logger = logging.getLogger(__name__)


class CopulaBB1(CopulaVirt):
    """BB1 (Joe-Clayton) copula — lower- AND upper-tail dependence.

    Parameters
    ----------
    tau_k : float
        Kendall's τ.  Must satisfy τ > 1 − 1/δ so that θ > 0.
    delta : float, optional
        Shape parameter δ ≥ 1 (default 1.5 if omitted).
        δ = 1 reduces to Clayton; δ → ∞ makes the upper tail dominate.
    """

    n_params: int = 2

    def __init__(self, **kwargs):
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

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
    # Shared sub-expressions
    # ------------------------------------------------------------------

    def _components(self, u: float, v: float):
        """A = u^{-θ}−1,  B = v^{-θ}−1,  S = (A^δ+B^δ)^{1/δ}.

        Returns (A, B, S, ok).  ok=False when any intermediate is ≤ 0.
        """
        th, de = self.theta, self.delta
        A = u ** (-th) - 1.0
        B = v ** (-th) - 1.0
        if A <= 0.0 or B <= 0.0:
            return None, None, None, False
        P = A ** de + B ** de
        if P <= 0.0:
            return None, None, None, False
        S = P ** (1.0 / de)
        return A, B, S, True

    # ------------------------------------------------------------------
    # CDF / PDF / h-function
    # ------------------------------------------------------------------

    def cdf(self, uv):
        """C(u,v) = (1 + S)^{−1/θ}."""
        u = minmaxEPS(uv[0])
        v = minmaxEPS(uv[1])
        try:
            A, B, S, ok = self._components(u, v)
            if not ok:
                raise ValueError('non-positive intermediate')
            result = (1.0 + S) ** (-1.0 / self.theta)
            if not np.isfinite(result):
                raise ValueError(f'non-finite: {result}')
            return float(np.clip(result, 0.0, 1.0))
        except (ZeroDivisionError, ValueError, OverflowError) as exc:
            logger.debug(
                'BB1.cdf: %s (θ=%.3f, δ=%.3f, u=%.3e, v=%.3e) → fallback EPS',
                exc, self.theta, self.delta, u, v,
            )
            return float(EPS)

    def cdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised closed-form CDF (BB1 has no statsmodels backend)."""
        uv = np.asarray(uv, dtype=float)
        u  = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v  = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        th, de = self.theta, self.delta
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            A = u ** (-th) - 1.0
            B = v ** (-th) - 1.0
            P = A ** de + B ** de
            S = P ** (1.0 / de)
            result = (1.0 + S) ** (-1.0 / th)
        return np.clip(np.where(np.isfinite(result), result, EPS), 0.0, 1.0)

    def pdf(self, uv):
        """c(u,v) = u^{-θ-1}·v^{-θ-1}·A^{δ-1}·B^{δ-1}·S^{1-2δ}·(1+S)^{-1/θ-2}·[θ(δ-1)+(θδ+1)S]."""
        u = minmaxEPS(uv[0])
        v = minmaxEPS(uv[1])
        try:
            A, B, S, ok = self._components(u, v)
            if not ok:
                raise ValueError('non-positive intermediate')
            th, de = self.theta, self.delta
            result = (
                u ** (-th - 1.0)
                * v ** (-th - 1.0)
                * A ** (de - 1.0)
                * B ** (de - 1.0)
                * S ** (1.0 - 2.0 * de)
                * (1.0 + S) ** (-1.0 / th - 2.0)
                * (th * (de - 1.0) + (th * de + 1.0) * S)
            )
            if not (np.isfinite(result) and result > 0.0):
                raise ValueError(f'non-finite or non-positive: {result}')
            return float(result)
        except (ZeroDivisionError, ValueError, OverflowError) as exc:
            logger.debug(
                'BB1.pdf: %s (θ=%.3f, δ=%.3f, u=%.3e, v=%.3e) → fallback EPS',
                exc, self.theta, self.delta, u, v,
            )
            return float(EPS)

    def pdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised closed-form PDF (BB1 has no statsmodels backend)."""
        uv = np.asarray(uv, dtype=float)
        u = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        th, de = self.theta, self.delta
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            A = u ** (-th) - 1.0
            B = v ** (-th) - 1.0
            P = A ** de + B ** de
            S = P ** (1.0 / de)
            result = (
                u ** (-th - 1.0)
                * v ** (-th - 1.0)
                * A ** (de - 1.0)
                * B ** (de - 1.0)
                * S ** (1.0 - 2.0 * de)
                * (1.0 + S) ** (-1.0 / th - 2.0)
                * (th * (de - 1.0) + (th * de + 1.0) * S)
            )
        return np.where(np.isfinite(result) & (result > 0.0), result, EPS)

    def logpdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Native log-PDF — bypasses ``log(max(pdf, EPS))``.

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
        th, de = self.theta, self.delta
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            A = np.maximum(u ** (-th) - 1.0, EPS)
            B = np.maximum(v ** (-th) - 1.0, EPS)
            P = A ** de + B ** de
            S = P ** (1.0 / de)
            return ((-th - 1.0) * (np.log(u) + np.log(v))
                    + (de - 1.0) * (np.log(A) + np.log(B))
                    + (1.0 - 2.0 * de) * np.log(np.maximum(S, EPS))
                    + (-1.0 / th - 2.0) * np.log1p(S)
                    + np.log(np.maximum(th * (de - 1.0) + (th * de + 1.0) * S, EPS)))

    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = u^{-θ-1} · A^{δ-1} · S^{1-δ} · (1+S)^{-1/θ-1}."""
        u = minmaxEPS(u)
        v = minmaxEPS(v)
        try:
            A, B, S, ok = self._components(u, v)
            if not ok:
                raise ValueError('non-positive intermediate')
            th, de = self.theta, self.delta
            result = (
                u ** (-th - 1.0)
                * A ** (de - 1.0)
                * S ** (1.0 - de)
                * (1.0 + S) ** (-1.0 / th - 1.0)
            )
            if not np.isfinite(result):
                raise ValueError(f'non-finite: {result}')
            return float(np.clip(result, 0.0, 1.0))
        except (ZeroDivisionError, ValueError, OverflowError) as exc:
            fallback = 1.0 if v > 0.5 else 0.0
            logger.debug(
                'BB1.h: %s (θ=%.3f, δ=%.3f, u=%.3e, v=%.3e) → fallback %.1f',
                exc, self.theta, self.delta, u, v, fallback,
            )
            return fallback

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
    def fit(cls, data: np.ndarray, method: str = 'mle') -> 'FitResult':
        """Fit BB1 by 2-D MLE over (θ, δ).

        ``method`` is accepted for API compatibility but MLE is always used
        (τ alone cannot identify both free parameters).

        Returns
        -------
        FitResult with copula fitted by MLE.
        """
        from scipy.stats import rankdata, kendalltau as _kendalltau

        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError(f'data must be shape (n, 2), got {data.shape}.')
        n = data.shape[0]
        if n < 8:
            raise ValueError(
                f'BB1 requires at least 8 observations for 2-D MLE, got {n}.'
            )

        uv = np.column_stack([
            rankdata(data[:, 0]) / (n + 1),
            rankdata(data[:, 1]) / (n + 1),
        ])

        if method == 'tau':
            logger.warning(
                "BB1.fit: method='tau' is under-determined (2 free params); "
                "falling back to MLE."
            )

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
                pdf_v = np.array([cop.pdf(list(row)) for row in uv])
                return -float(np.sum(np.log(np.maximum(pdf_v, EPS))))
            except Exception:
                return 1e15

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
        pdf_vals = np.array([cop.pdf(list(row)) for row in uv])
        log_lik  = float(np.sum(np.log(np.maximum(pdf_vals, EPS))))

        return FitResult(
            copula=cop,
            method='mle',
            tau_k=tau_k_hat,
            log_likelihood=log_lik,
            n_obs=n,
            uv=uv,
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
    from prg.copulas.archimedean.clayton import CopulaClayton
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
