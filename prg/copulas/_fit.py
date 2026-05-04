"""
prg.copulas._fit — fitting, goodness-of-fit, and diagnostic helpers.

Contents
--------
_eval_log_likelihood, _empirical_copula, _cvm_statistic, _empirical_tail_dep
    Pseudo-observation utilities (used by both `CopulaVirt.fit` and
    `FitResult.gof_test`).
GoFResult
    Cramér-von Mises GoF test result.
FitResult
    `CopulaVirt.fit()` return value: τ̂, log-likelihood, AIC/BIC/AICc/HQC,
    GoF test, bootstrap CI, K-fold CV, and diagnostic plots.

These were originally in ``_base.py``; extracted to keep the base module focused
on the copula class hierarchy.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from prg.plot_style import DEFAULT_FONT_SIZE as FONT_SIZE

if TYPE_CHECKING:
    from prg.copulas._base import CopulaVirt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pseudo-observation utilities
# ---------------------------------------------------------------------------

def _eval_log_likelihood(copula: "CopulaVirt", uv: np.ndarray) -> float:
    """∑ log c(û_i, v̂_i) on pseudo-observations.

    Uses the vectorised ``logpdf_array`` (50-100× faster than a Python loop
    over scalar ``pdf``).
    """
    return float(np.sum(copula.logpdf_array(np.asarray(uv, dtype=float))))


def _empirical_copula(uv: np.ndarray, points: np.ndarray) -> np.ndarray:
    """C_n(u,v) = (1/n) Σ 1{û_i ≤ u, v̂_i ≤ v}, evaluated at given points."""
    leq_u = uv[:, 0][None, :] <= points[:, 0][:, None]   # (m, n)
    leq_v = uv[:, 1][None, :] <= points[:, 1][:, None]
    return (leq_u & leq_v).mean(axis=1)


def _cvm_statistic(uv: np.ndarray, copula: "CopulaVirt") -> float:
    """Cramér-von Mises statistic S_n = Σ (C_n(u_i,v_i) − C_θ(u_i,v_i))²."""
    Cn     = _empirical_copula(uv, uv)
    Ctheta = copula.cdf_array(np.asarray(uv, dtype=float))
    return float(np.sum((Cn - Ctheta) ** 2))


def _empirical_tail_dep(uv: np.ndarray, u_grid: np.ndarray, side: str) -> np.ndarray:
    """Non-parametric tail dependence estimator on a grid of thresholds.

    side='lower': λ̂_L(u) = C_n(u, u) / u
    side='upper': λ̂_U(u) = (1 − 2u + C_n(u, u)) / (1 − u)
    Reference: Joe (1997); Caillault & Guégan (2009).
    """
    out = np.empty(len(u_grid))
    for k, u in enumerate(u_grid):
        Cn_uu = float(np.mean((uv[:, 0] <= u) & (uv[:, 1] <= u)))
        if side == 'lower':
            out[k] = Cn_uu / u if u > 0.0 else 0.0
        else:
            out[k] = (1.0 - 2.0*u + Cn_uu) / (1.0 - u) if u < 1.0 else 0.0
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# GoFResult
# ---------------------------------------------------------------------------

@dataclass
class GoFResult:
    """Returned by :meth:`FitResult.gof_test` — Cramér-von Mises GoF test."""
    statistic:        float
    p_value:          float
    B:                int
    n_valid_bootstrap: int
    bootstrap_stats:  np.ndarray

    def __repr__(self) -> str:
        return (f'GoFResult(S_n={self.statistic:.4f}, p-value={self.p_value:.3f}, '
                f'B={self.n_valid_bootstrap}/{self.B})')


# ---------------------------------------------------------------------------
# FitResult
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    """Returned by :meth:`CopulaVirt.fit`."""
    copula:         "CopulaVirt"
    method:         str
    tau_k:          float
    log_likelihood: float
    n_obs:          int
    uv:             np.ndarray   # pseudo-observations used for the fit, shape (n, 2)

    @property
    def n_params(self) -> int:
        return getattr(self.copula, 'n_params', 1)

    @property
    def aic(self) -> float:
        """Akaike Information Criterion: 2k − 2·loglik (smaller is better)."""
        return 2.0 * self.n_params - 2.0 * self.log_likelihood

    @property
    def bic(self) -> float:
        """Bayesian Information Criterion: k·log(n) − 2·loglik (smaller is better)."""
        return self.n_params * np.log(self.n_obs) - 2.0 * self.log_likelihood

    @property
    def aicc(self) -> float:
        """Corrected AIC for small samples: AIC + 2k(k+1)/(n−k−1)."""
        k, n = self.n_params, self.n_obs
        if n - k - 1 <= 0:
            return float('nan')
        return self.aic + 2.0 * k * (k + 1) / (n - k - 1)

    @property
    def hqc(self) -> float:
        """Hannan-Quinn: 2k·log(log n) − 2·loglik (asymptotically less biased than BIC)."""
        return 2.0 * self.n_params * np.log(np.log(self.n_obs)) - 2.0 * self.log_likelihood

    def __repr__(self) -> str:
        name = self.copula.copula_enum.value.LONG_NAME
        return (
            f'FitResult(copula={name!r}, method={self.method!r}, '
            f'tau_k={self.tau_k:.4f}, loglik={self.log_likelihood:.4f}, '
            f'AIC={self.aic:.2f}, BIC={self.bic:.2f}, n={self.n_obs})'
        )

    # ------------------------------------------------------------------
    # Goodness-of-fit (Cramér-von Mises, parametric bootstrap)
    # ------------------------------------------------------------------
    def gof_test(self, B: int = 100, seed: int | None = None) -> GoFResult:
        """Cramér-von Mises GoF test via parametric bootstrap.

        H₀: data was generated by the fitted copula family.
        Returns S_n and a bootstrap p-value over B replicates.
        Reference: Genest, Rémillard & Beaudoin (2009).
        """
        try:
            self.copula.cdf([0.5, 0.5])
        except NotImplementedError:
            raise NotImplementedError(
                f'GoF test requires C(u,v); not implemented for '
                f'{self.copula.__class__.__name__}.'
            )
        rng = np.random.default_rng(seed)
        S_n = _cvm_statistic(self.uv, self.copula)

        cls   = self.copula.__class__
        stats = np.full(B, np.nan)
        # Log every ~10 % of progress (B/10 iterations) for long bootstraps.
        log_every = max(B // 10, 1)
        for b in range(B):
            sample = self.copula.sample(n=self.n_obs,
                                        seed=int(rng.integers(0, 2**31 - 1)))
            try:
                r_b      = cls.fit(sample, method=self.method)
                stats[b] = _cvm_statistic(r_b.uv, r_b.copula)
            except Exception as e:
                logger.debug('Bootstrap iter %d failed: %s', b, e)
            if (b + 1) % log_every == 0:
                logger.info('GoF bootstrap: %d / %d done', b + 1, B)

        valid    = ~np.isnan(stats)
        n_valid  = int(valid.sum())
        p_value  = float(np.mean(stats[valid] >= S_n)) if n_valid > 0 else float('nan')
        return GoFResult(statistic=S_n, p_value=p_value, B=B,
                         n_valid_bootstrap=n_valid,
                         bootstrap_stats=stats[valid])

    # ------------------------------------------------------------------
    # Bootstrap confidence interval on tau_k
    # ------------------------------------------------------------------
    def bootstrap_ci(self, B: int = 500, alpha: float = 0.05,
                     seed: int | None = None) -> tuple[float, float]:
        """Non-parametric bootstrap percentile CI for tau_k at level (1-alpha).

        Resamples the pseudo-observations with replacement B times, refits with
        the same method, and returns the (alpha/2, 1-alpha/2) quantiles.
        """
        rng = np.random.default_rng(seed)
        cls = self.copula.__class__
        tau_boots = np.full(B, np.nan)
        log_every = max(B // 10, 1)
        for b in range(B):
            idx    = rng.integers(0, self.n_obs, size=self.n_obs)
            sample = self.uv[idx]
            try:
                tau_boots[b] = cls.fit(sample, method=self.method).tau_k
            except Exception as e:
                logger.debug('Bootstrap iter %d failed: %s', b, e)
            if (b + 1) % log_every == 0:
                logger.info('Bootstrap τ-CI: %d / %d done', b + 1, B)
        valid = ~np.isnan(tau_boots)
        if valid.sum() < 10:
            raise RuntimeError(f'Bootstrap failed: only {valid.sum()}/{B} valid replicates.')
        lo = float(np.quantile(tau_boots[valid], alpha / 2.0))
        hi = float(np.quantile(tau_boots[valid], 1.0 - alpha / 2.0))
        return lo, hi

    # ------------------------------------------------------------------
    # K-fold cross-validation log-likelihood
    # ------------------------------------------------------------------
    def cv_loglik(self, K: int = 5, seed: int | None = None) -> float:
        """K-fold CV log-likelihood (held-out test loglik summed over folds).

        For each fold, the copula is refitted on the K−1 training folds (using
        the same `method`) and its log-density is summed over the held-out
        test fold's pseudo-observations. Higher is better.
        """
        if K < 2 or K > self.n_obs:
            raise ValueError(f'K must be in [2, n_obs], got {K} for n={self.n_obs}.')
        rng = np.random.default_rng(seed)
        idx = rng.permutation(self.n_obs)
        folds = np.array_split(idx, K)
        cls = self.copula.__class__

        cv_ll = 0.0
        for k in range(K):
            test_idx  = folds[k]
            train_idx = np.concatenate([folds[j] for j in range(K) if j != k])
            try:
                r_k    = cls.fit(self.uv[train_idx], method=self.method)
                cv_ll += _eval_log_likelihood(r_k.copula, self.uv[test_idx])
            except Exception as e:
                logger.warning('CV fold %d failed: %s', k, e)
                return float('nan')
        return float(cv_ll)

    # ------------------------------------------------------------------
    # Visual diagnostics
    # ------------------------------------------------------------------
    def plot_diagnostics(self, plot_dir: str, prefix: str = '') -> None:
        """6-panel diagnostic plot:
        (0,0) pseudo-observations + fitted PDF contours
        (0,1) PP plot  C_n vs C_θ at the data points
        (0,2) lower tail dependence  λ̂_L(u) vs fitted λ_L
        (1,0) empirical copula heatmap C_n(u,v)
        (1,1) residuals heatmap C_n − C_θ
        (1,2) upper tail dependence  λ̂_U(u) vs fitted λ_U
        """
        try:
            self.copula.cdf([0.5, 0.5])
            has_cdf = True
        except NotImplementedError:
            has_cdf = False

        name      = self.copula.copula_enum.value.LONG_NAME
        short     = self.copula.copula_enum.value.SHORT_NAME
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(
            f'{name} — fit diagnostics  (τ̂={self.tau_k:.3f}, n={self.n_obs})',
            fontsize=FONT_SIZE + 1, y=0.995,
        )

        grid    = np.linspace(0.01, 0.99, 80)
        GX, GY  = np.meshgrid(grid, grid)
        Z_pdf   = np.vectorize(lambda u, v: self.copula.pdf([u, v]))(GX, GY)
        vmax    = float(np.nanpercentile(Z_pdf, 95))

        # (0,0) — pseudo-obs over fitted PDF contours
        ax = axes[0, 0]
        ax.contourf(GX, GY, Z_pdf, levels=10, cmap='Blues', alpha=0.4,
                    vmin=0, vmax=max(vmax, 1e-10))
        ax.scatter(self.uv[:, 0], self.uv[:, 1], s=4, alpha=0.55, c='black')
        ax.set_title('Pseudo-obs + fitted PDF')
        ax.set_xlabel('u'); ax.set_ylabel('v'); ax.set_aspect('equal')

        if has_cdf:
            Cn_pts = _empirical_copula(self.uv, self.uv)
            Ct_pts = np.array([self.copula.cdf(list(row)) for row in self.uv])
            points = np.column_stack([GX.ravel(), GY.ravel()])
            Cn_g   = _empirical_copula(self.uv, points).reshape(GX.shape)
            Ct_g   = np.vectorize(lambda u, v: self.copula.cdf([u, v]))(GX, GY)

            # (0,1) — PP plot
            ax = axes[0, 1]
            ax.scatter(Cn_pts, Ct_pts, s=6, alpha=0.55)
            ax.plot([0, 1], [0, 1], 'r--', lw=1)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
            ax.set_xlabel('$C_n$ (empirical)'); ax.set_ylabel(r'$C_\theta$ (fitted)')
            ax.set_title('PP plot')

            # (1,0) — empirical copula
            ax = axes[1, 0]
            cs = ax.contourf(GX, GY, Cn_g, levels=11, cmap='viridis', vmin=0, vmax=1)
            fig.colorbar(cs, ax=ax)
            ax.set_title(r'$C_n(u,v)$ (empirical)')
            ax.set_xlabel('u'); ax.set_ylabel('v'); ax.set_aspect('equal')

            # (1,1) — residuals
            ax    = axes[1, 1]
            diff  = Cn_g - Ct_g
            vmaxd = float(max(abs(diff.min()), abs(diff.max()), 1e-3))
            cs = ax.contourf(GX, GY, diff, levels=15, cmap='RdBu_r',
                             vmin=-vmaxd, vmax=vmaxd)
            fig.colorbar(cs, ax=ax)
            ax.set_title(rf'Residuals $C_n - C_\theta$  (max |Δ|={vmaxd:.3f})')
            ax.set_xlabel('u'); ax.set_ylabel('v'); ax.set_aspect('equal')
        else:
            for r, c in [(0, 1), (1, 0), (1, 1)]:
                axes[r, c].axis('off')
                axes[r, c].text(0.5, 0.5, f'CDF not available\nfor {name}',
                                ha='center', va='center', transform=axes[r, c].transAxes)

        # (0,2) — lower tail dependence
        u_low  = np.linspace(0.02, 0.30, 25)
        u_high = np.linspace(0.70, 0.98, 25)
        lL_emp = _empirical_tail_dep(self.uv, u_low,  'lower')
        lU_emp = _empirical_tail_dep(self.uv, u_high, 'upper')
        lL_th, lU_th = self.copula.tail_dependence()

        ax = axes[0, 2]
        ax.plot(u_low, lL_emp, 'o-', ms=4, label=r'Empirical $\hat\lambda_L(u)$')
        if not np.isnan(lL_th):
            ax.axhline(lL_th, color='red', ls='--', lw=1.2,
                       label=rf'Fitted $\lambda_L = {lL_th:.3f}$')
        ax.set_xlabel('u'); ax.set_ylabel(r'$\lambda_L$')
        ax.set_title('Lower tail dependence')
        ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=9)

        # (1,2) — upper tail dependence
        ax = axes[1, 2]
        ax.plot(u_high, lU_emp, 'o-', ms=4, label=r'Empirical $\hat\lambda_U(u)$')
        if not np.isnan(lU_th):
            ax.axhline(lU_th, color='red', ls='--', lw=1.2,
                       label=rf'Fitted $\lambda_U = {lU_th:.3f}$')
        ax.set_xlabel('u'); ax.set_ylabel(r'$\lambda_U$')
        ax.set_title('Upper tail dependence')
        ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{prefix}Diagnostics_{short}.png'),
                    bbox_inches='tight')
        plt.close()
