"""
Student-t copula with free degrees-of-freedom ν.

Parameters
----------
tau_k : float
    Kendall's τ ∈ (−1, 1).  Linked to the correlation via ρ = sin(πτ/2).
df : float, optional
    Degrees of freedom ν > 2 (default 4.0).
    ν → ∞  recovers the Gaussian copula (zero tail dependence).
    Smaller ν → heavier symmetric tails.

Relationships
-------------
    ρ(τ_K) = sin(π·τ/2)
    λ_L = λ_U = 2·t_{ν+1}(−√((ν+1)(1−ρ)/(1+ρ)))   [symmetric tail dependence]

Fitting
-------
    n_params = 2 — MLE jointly over (ρ, ν) via L-BFGS-B.
    method='tau' is accepted for API compatibility and warns before falling
    back to MLE (τ alone cannot identify both ρ and ν).

Notes
-----
    CDF has no closed form (requires bivariate t integral) — plot_cdf is
    silently skipped and the GoF test based on CDF is not available.
"""
if __name__ == '__main__':
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import logging
import math
import numpy as np
from scipy.stats import t as _t
from scipy.optimize import minimize
from statsmodels.distributions.copula.api import StudentTCopula

from prg.copulas._base import CopulaVirt, FitResult
from prg.exceptions    import CopulaParameterError
from prg.tools.tools   import EPS, ONE_MINUS_EPS, EPS_MINUS_ONE, minmaxEPS

logger = logging.getLogger(__name__)


class CopulaStudent(CopulaVirt):
    """Student-t copula with free degrees of freedom ν.

    Parameters
    ----------
    tau_k : float
        Kendall's τ.  Full range (−1, 1).
    df : float, optional
        Degrees of freedom ν > 2 (default 4.0 if omitted).
        ν → ∞ → Gaussian copula; small ν → heavy symmetric tails.
    """

    n_params: int = 2

    def __init__(self, **kwargs):
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    def _update_params(self):
        df = float(self.params.get('df', 4.0))
        if df <= 2.0:
            raise CopulaParameterError(
                f'Student: df must be > 2, got {df:.4f}.'
            )
        self.df    = df
        self.theta = math.sin(self.params['tau_k'] * math.pi / 2.0)
        self.theta = max(EPS_MINUS_ONE, min(ONE_MINUS_EPS, self.theta))
        corr = np.array([[1., self.theta], [self.theta, 1.]])
        self._model = StudentTCopula(corr=corr, df=self.df)
        # Persist df back so plot_multi_tau, bootstrap, GoF round-trips see it
        self.params['df'] = df

    # ------------------------------------------------------------------
    # PDF / CDF / h-function
    # ------------------------------------------------------------------

    def pdf(self, uv):
        u = minmaxEPS(uv[0])
        v = minmaxEPS(uv[1])
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            result = float(self._model.pdf([u, v]))
        if not (np.isfinite(result) and result > 0.0):
            logger.debug(
                'Student.pdf: result=%s (θ=%.3f, df=%.2f, u=%.3e, v=%.3e) → fallback EPS',
                result, self.theta, self.df, u, v,
            )
            return float(EPS)
        return result

    def cdf(self, uv):
        raise NotImplementedError(
            'CopulaStudent: CDF has no closed form (bivariate Student-t integral).'
        )

    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = t_{ν+1}((t_ν⁻¹(v) − ρ·t_ν⁻¹(u)) · √((ν+1)/((ν+(t_ν⁻¹(u))²)(1−ρ²))))."""
        x_u = _t.ppf(minmaxEPS(u), df=self.df)
        x_v = _t.ppf(minmaxEPS(v), df=self.df)
        scale = math.sqrt(
            (self.df + 1.0) / ((self.df + x_u ** 2) * (1.0 - self.theta ** 2))
        )
        return float(_t.cdf((x_v - self.theta * x_u) * scale, df=self.df + 1))

    def tail_dependence(self) -> tuple[float, float]:
        """Symmetric tail dependence: λ = 2·t_{ν+1}(−√((ν+1)(1−ρ)/(1+ρ)))."""
        arg = -math.sqrt(
            (self.df + 1.0) * (1.0 - self.theta) / (1.0 + self.theta)
        )
        lam = 2.0 * float(_t.cdf(arg, df=self.df + 1))
        return lam, lam

    # ------------------------------------------------------------------
    # 2-D MLE fitting (overrides CopulaVirt.fit)
    # ------------------------------------------------------------------

    @classmethod
    def fit(cls, data: np.ndarray, method: str = 'mle') -> 'FitResult':
        """Fit the Student-t copula by 2-D MLE over (ρ, ν).

        ``method`` is accepted for API compatibility.  If 'tau' is passed,
        a warning is issued and MLE is used (τ alone under-determines ν).

        Parameters
        ----------
        data   : array-like, shape (n, 2)
        method : {'mle', 'tau'}  (only 'mle' is meaningful; 'tau' warns)

        Returns
        -------
        FitResult with method='mle'.
        """
        from scipy.stats import rankdata, kendalltau as _kendalltau

        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError(f'data must be shape (n, 2), got {data.shape}.')
        n = data.shape[0]
        if n < 8:
            raise ValueError(
                f'Student requires at least 8 observations for 2-D MLE, got {n}.'
            )

        uv = np.column_stack([
            rankdata(data[:, 0]) / (n + 1),
            rankdata(data[:, 1]) / (n + 1),
        ])

        if method == 'tau':
            logger.warning(
                "Student.fit: method='tau' is under-determined (2 free params ρ and ν); "
                "falling back to MLE."
            )

        # Initial point: empirical τ → ρ₀ via sin formula, ν₀ = 4
        tau_emp, _ = _kendalltau(data[:, 0], data[:, 1])
        tau_emp    = float(np.clip(tau_emp, -0.99, 0.99))
        rho0       = math.sin(math.pi * tau_emp / 2.0)
        rho0       = float(np.clip(rho0, -0.98, 0.98))
        df0        = 4.0

        def _neg_ll(x: np.ndarray) -> float:
            rho = float(np.clip(x[0], EPS_MINUS_ONE + 1e-9, ONE_MINUS_EPS - 1e-9))
            df  = float(x[1])
            if df <= 2.0:
                return 1e15
            # Convert ρ → τ to construct the copula
            tau_t = float(np.clip((2.0 / math.pi) * math.asin(rho),
                                  EPS_MINUS_ONE, ONE_MINUS_EPS))
            try:
                cop    = cls(tau_k=tau_t, df=df)
                pdf_v  = np.array([cop.pdf(list(row)) for row in uv])
                return -float(np.sum(np.log(np.maximum(pdf_v, EPS))))
            except Exception:
                return 1e15

        res = minimize(
            _neg_ll,
            x0=[rho0, df0],
            method='L-BFGS-B',
            bounds=[
                (EPS_MINUS_ONE + 1e-6, ONE_MINUS_EPS - 1e-6),
                (2.0 + 1e-4, None),
            ],
            options={'ftol': 1e-10, 'gtol': 1e-8, 'maxiter': 400},
        )

        rho_hat = float(np.clip(res.x[0], EPS_MINUS_ONE + 1e-9, ONE_MINUS_EPS - 1e-9))
        df_hat  = float(max(2.0 + 1e-4, res.x[1]))
        tau_hat = float(np.clip((2.0 / math.pi) * math.asin(rho_hat),
                                EPS_MINUS_ONE, ONE_MINUS_EPS))

        cop      = cls(tau_k=tau_hat, df=df_hat)
        pdf_vals = np.array([cop.pdf(list(row)) for row in uv])
        log_lik  = float(np.sum(np.log(np.maximum(pdf_vals, EPS))))

        return FitResult(
            copula=cop,
            method='mle',
            tau_k=tau_hat,
            log_likelihood=log_lik,
            n_obs=n,
            uv=uv,
        )


if __name__ == '__main__':
    from pathlib import Path
    import numpy as np

    cop = CopulaStudent(tau_k=0.5)
    lL, lU = cop.tail_dependence()
    print(f'Copula   : {cop.copula_enum.value.LONG_NAME}')
    print(f'tau_k    : {cop.params["tau_k"]:.4f}  range={cop.tau_range}')
    print(f'theta(ρ) : {cop.theta:.6f}  [= sin(π·τ/2)]')
    print(f'df (ν)   : {cop.df:.2f}  (default 4.0)')
    print(f'pdf(0.3, 0.7) = {cop.pdf([0.3, 0.7]):.6f}')
    print(f'h(0.7 | 0.3)  = {cop.conditional_cdf(0.7, 0.3):.6f}')
    print(f'tail dep : λ_L = λ_U = {lL:.4f}  [symmetric, heavier than Gaussian]')
    print(f'(CDF has no closed form — plot_cdf silently skipped)')

    # Test with custom df
    cop8 = CopulaStudent(tau_k=0.5, df=8.0)
    lL8, lU8 = cop8.tail_dependence()
    print(f'\ndf=8: λ_L = λ_U = {lL8:.4f}  [lighter tails than df=4]')

    # Simulate data and fit
    rng = np.random.default_rng(42)
    data = cop.sample(500, seed=42)
    res = CopulaStudent.fit(data, method='mle')
    print(f'\nFit result: {res}')

    plot_dir = Path('./data/Plots/Copulas')
    plot_dir.mkdir(parents=True, exist_ok=True)
    cop.plot_pdf(plot_dir)
    cop.plot_cdf(plot_dir)   # skipped gracefully (NotImplementedError)
    cop.plot_h_function(plot_dir)
    cop.plot_samples(plot_dir)
    cop.plot_overview(plot_dir)
    cop.plot_multi_tau(plot_dir)
