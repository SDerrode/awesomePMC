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
    h(v|u) = t_{ν+1}((t_ν⁻¹(v) − ρ t_ν⁻¹(u)) · √((ν+1)/((ν + t_ν⁻¹(u)²)(1−ρ²))))

Numerics
--------
The log-density is native — the bivariate t log-density minus the two
univariate ones (Demarta & McNeil 2005), with x = t_ν⁻¹(u), y = t_ν⁻¹(v):

    log c = log a + 2[log B(a, ½) − ½ log π] − ½ log(1 − ρ²)
            − ((ν+2)/2) · log1p(Q / (ν(1 − ρ²)))
            + ((ν+1)/2) · [log1p(x²/ν) + log1p(y²/ν)],       a = ν/2,

the Γ-ratio Γ((ν+2)/2)Γ(ν/2)/Γ((ν+1)/2)² being written through the Beta
function so that it does not cancel for large ν. The quadratic form
Q = x² − 2ρxy + y² is taken as (x − σy)² + 2(1 − |ρ|)|xy| when ρxy > 0
(σ = sign ρ) — no cancellation near the diagonal in strong dependence — and
log1p(x²/ν) as 2 log|x| − log ν + log1p(ν/x²) when x² > ν (technique 12 of
the numerics notes). It replaces the statsmodels backend, whose density was
floored at ``EPS``: at ν = 30, τ = 0.95 every tail point read log c = −36.04
while the true values reach −80 (audit RB-4). ``pdf`` and ``pdf_array`` are
``exp`` of it, with no floor.

Quantiles use ``stdtrit(ν, u)`` for u ≤ ½ and ``−stdtrit(ν, 1 − u)`` above;
h is ``stdtr(ν+1, z)`` (an incomplete-Beta ratio, relatively accurate in the
lower tail) with √(ν + x²) = hypot(√ν, x), and ``inv_h`` is the closed-form
inverse v = t_ν(z·hypot(√ν, x)·√((1−ρ²)/(ν+1)) + ρx), z = t_{ν+1}⁻¹(w)
(Aas et al. 2009), replacing the numerical Brent inversion of the base class.

Reference: Demarta, S. & McNeil, A. J. (2005). The t copula and related
copulas. *International Statistical Review* 73(1), 111–129,
doi:10.1111/j.1751-5823.2005.tb00254.x (density, τ = (2/π) arcsin ρ and the
tail-dependence coefficient above); Aas, K., Czado, C., Frigessi, A. &
Bakken, H. (2009). Pair-copula constructions of multiple dependence.
*Insurance: Mathematics and Economics* 44(2), 182–198,
doi:10.1016/j.insmatheco.2007.02.001 (h-function and its inverse).

Fitting
-------
    n_params = 2 — MLE jointly over (ρ, ν) via L-BFGS-B, with ν bounded by
    ``EXTRA_PARAM_BOUNDS_BY_PARAM["df"]`` of ``pmcprg.copulas._base`` — the same
    bounds ICE uses (audit A-2, RB-4).
    method='tau' is accepted for API compatibility and warns before falling
    back to MLE (τ alone cannot identify both ρ and ν).

Notes
-----
    CDF has no closed form (requires bivariate t integral) — plot_cdf is
    silently skipped and the GoF test based on CDF is not available.
"""
if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import logging
import math
import numpy as np
from scipy.special import betaln, stdtr, stdtrit
from scipy.stats import t as _t
from scipy.optimize import minimize

from pmcprg.copulas._base import CopulaVirt, FitResult, EXTRA_PARAM_BOUNDS_BY_PARAM
from pmcprg.exceptions    import CopulaParameterError
from pmcprg.numerics   import EPS, MIN_POSITIVE, ONE_MINUS_EPS, EPS_MINUS_ONE, minmaxEPS

logger = logging.getLogger(__name__)


def _t_quantile(df: float, p):
    """t_ν⁻¹(p): ``stdtrit(ν, p)`` for p ≤ ½, ``−stdtrit(ν, 1 − p)`` above."""
    p = np.asarray(p, dtype=float)
    return np.where(p <= 0.5, stdtrit(df, p), -stdtrit(df, 1.0 - p))


def _log1p_sq_over(x, nu: float):
    """log1p(x²/ν), as 2 log|x| − log ν + log1p(ν/x²) when x² > ν."""
    x2 = x * x
    big = x2 > nu
    with np.errstate(divide='ignore'):
        return np.where(big,
                        2.0 * np.log(np.abs(x)) - math.log(nu) + np.log1p(nu / np.where(big, x2, 1.0)),
                        np.log1p(x2 / nu))


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
        r = abs(self.theta)
        self._one_minus_rho2 = (1.0 - r) * (1.0 + r)
        a = 0.5 * df
        self._logc_const = (math.log(a) + 2.0 * (betaln(a, 0.5) - 0.5 * math.log(math.pi))
                            - 0.5 * (math.log1p(-r) + math.log1p(r)))
        # Persist df back so plot_multi_tau, bootstrap, GoF round-trips see it
        self.params['df'] = df

    # ------------------------------------------------------------------
    # PDF / CDF / h-function
    # ------------------------------------------------------------------

    def logpdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Native log-density (module docstring) — no floor."""
        uv = np.asarray(uv, dtype=float)
        u = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        nu = self.df
        x = _t_quantile(nu, u)
        y = _t_quantile(nu, v)
        r = abs(self.theta)
        sg = 1.0 if self.theta >= 0.0 else -1.0
        sxy = sg * x * y                                   # ≥ 0 ⇔ ρxy ≥ 0
        Q = np.where(sxy > 0.0,
                     (x - sg * y) ** 2 + 2.0 * (1.0 - r) * sxy,
                     x * x + y * y - 2.0 * r * sxy)
        return (self._logc_const
                - 0.5 * (nu + 2.0) * np.log1p(Q / (nu * self._one_minus_rho2))
                + 0.5 * (nu + 1.0) * (_log1p_sq_over(x, nu) + _log1p_sq_over(y, nu)))

    def pdf_array(self, uv: np.ndarray) -> np.ndarray:
        """``exp`` of :meth:`logpdf_array` — no floor."""
        with np.errstate(under='ignore', over='ignore'):
            return np.exp(self.logpdf_array(uv))

    def pdf(self, uv):
        """``exp`` of the native log-density — no floor."""
        u = minmaxEPS(uv[0])
        v = minmaxEPS(uv[1])
        return float(np.exp(self.logpdf_array(np.array([[u, v]]))[0]))

    def cdf(self, uv):
        raise NotImplementedError(
            'CopulaStudent: CDF has no closed form (bivariate Student-t integral).'
        )

    def _h_array(self, v: np.ndarray, u: np.ndarray) -> np.ndarray:
        nu = self.df
        x = _t_quantile(nu, u)
        y = _t_quantile(nu, v)
        z = ((y - self.theta * x) * math.sqrt((nu + 1.0) / self._one_minus_rho2)
             / np.hypot(math.sqrt(nu), x))
        return stdtr(nu + 1.0, z)

    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = t_{ν+1}((t_ν⁻¹(v) − ρ·t_ν⁻¹(u)) · √((ν+1)/((ν+(t_ν⁻¹(u))²)(1−ρ²))))."""
        return float(self._h_array(np.array([minmaxEPS(v)]), np.array([minmaxEPS(u)]))[0])

    def inv_h(self, w: float, u: float) -> float:
        """Closed-form inverse of h(v|u) (module docstring)."""
        return float(self.inv_h_array(np.array([w]), np.array([u]))[0])

    def inv_h_array(self, w: np.ndarray, u: np.ndarray) -> np.ndarray:
        """v = t_ν(t_{ν+1}⁻¹(w)·√((ν + x²)(1−ρ²)/(ν+1)) + ρx), x = t_ν⁻¹(u).

        w is clipped to [MIN_POSITIVE, 1 − EPS] only, so an h-value below EPS
        still inverts to its own v; u and the result to [EPS, 1 − EPS].
        """
        w = np.clip(np.asarray(w, dtype=float), MIN_POSITIVE, ONE_MINUS_EPS)
        u = np.clip(np.asarray(u, dtype=float), EPS, ONE_MINUS_EPS)
        nu = self.df
        x = _t_quantile(nu, u)
        z = _t_quantile(nu + 1.0, w)
        y = (z * np.hypot(math.sqrt(nu), x) * math.sqrt(self._one_minus_rho2 / (nu + 1.0))
             + self.theta * x)
        return np.clip(stdtr(nu, y), EPS, ONE_MINUS_EPS)

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
        ν is bounded — and started — by ``EXTRA_PARAM_BOUNDS_BY_PARAM["df"]``,
        the bounds ICE uses (audit RB-4); the objective is the native
        log-density, without floor.

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

        df_lo, df_hi, df0 = EXTRA_PARAM_BOUNDS_BY_PARAM["df"]

        # Initial point: empirical τ → ρ₀ via sin formula, ν₀ from the registry
        tau_emp, _ = _kendalltau(data[:, 0], data[:, 1])
        tau_emp    = float(np.clip(tau_emp, -0.99, 0.99))
        rho0       = math.sin(math.pi * tau_emp / 2.0)
        rho0       = float(np.clip(rho0, -0.98, 0.98))

        def _neg_ll(x: np.ndarray) -> float:
            rho = float(np.clip(x[0], EPS_MINUS_ONE + 1e-9, ONE_MINUS_EPS - 1e-9))
            df  = float(x[1])
            # Convert ρ → τ to construct the copula
            tau_t = float(np.clip((2.0 / math.pi) * math.asin(rho),
                                  EPS_MINUS_ONE, ONE_MINUS_EPS))
            try:
                ll = float(np.sum(cls(tau_k=tau_t, df=df).logpdf_array(uv)))
            except CopulaParameterError:
                return 1e15                      # ν outside the admissible ν > 2
            # An optimiser cannot use NaN/−inf; a large finite value keeps
            # L-BFGS-B away from the point (audit RB-7).
            return -ll if np.isfinite(ll) else 1e15

        res = minimize(
            _neg_ll,
            x0=[rho0, df0],
            method='L-BFGS-B',
            bounds=[
                (EPS_MINUS_ONE + 1e-6, ONE_MINUS_EPS - 1e-6),
                (df_lo, df_hi),
            ],
            options={'ftol': 1e-10, 'gtol': 1e-8, 'maxiter': 400},
        )

        rho_hat = float(np.clip(res.x[0], EPS_MINUS_ONE + 1e-9, ONE_MINUS_EPS - 1e-9))
        df_hat  = float(np.clip(res.x[1], df_lo, df_hi))
        tau_hat = float(np.clip((2.0 / math.pi) * math.asin(rho_hat),
                                EPS_MINUS_ONE, ONE_MINUS_EPS))

        cop     = cls(tau_k=tau_hat, df=df_hat)
        log_lik = float(np.sum(cop.logpdf_array(uv)))

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
    print('(CDF has no closed form — plot_cdf silently skipped)')

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
