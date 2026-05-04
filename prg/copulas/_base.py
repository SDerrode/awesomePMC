if __name__ == "__main__":
    import sys, pathlib

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import logging
import os
import matplotlib.pyplot as plt
import numpy as np

from enum import Enum, unique
from dataclasses import dataclass, field

from prg.tools.tools import EPS, ONE_MINUS_EPS, EPS_MINUS_ONE, minmaxEPS
from prg.settings.plot_settings import facecolor, dpi, BIGGER_SIZE
from prg.exceptions import CopulaParameterError, CopulaNotAvailableError

# AMH τ_K range constants (computed at import time, θ ∈ [−1, 1))
_AMH_TAU_MIN = float(1.0 - 2.0 * (4.0 * np.log(2.0) - 1.0) / 3.0)  # τ(θ=−1) ≈ −0.1817
_AMH_TAU_MAX = float(1.0 / 3.0 - EPS)                                  # limit τ → 1/3

logger = logging.getLogger(__name__)


@dataclass
class CopulaDataMixin:
    ID: int
    SHORT_NAME: str
    LONG_NAME: str
    CLASS_NAME: str
    AVAILABLE: bool
    PARAMETERS_SET_NAME: list[str] = field(default_factory=list)
    TAU_MIN_MAX: list[float] = field(default_factory=list)
    MODULE: str = ""   # dotted import path, e.g. "prg.copulas.elliptical.gaussian"


@unique
class CopulaEnum(CopulaDataMixin, Enum):
    PRODUCT  = 1,  "Prod",    "Product",              "CopulaProduct",  True,  ["tau_k"], [0.0, 0.0],                    "prg.copulas.explicit.product"
    GAUSSIAN = 2,  "Gauss",   "Gaussian",             "CopulaGaussian", True,  ["tau_k"], [-1.0, 1.0],                   "prg.copulas.elliptical.gaussian"
    STUDENT  = 3,  "Student", "Student",              "CopulaStudent",  True,  ["tau_k"], [-1.0, 1.0],                   "prg.copulas.elliptical.student"
    GH       = 4,  "GH",      "Gumbel-Hougaard",      "CopulaGH",       True,  ["tau_k"], [0.0 + EPS, 1.0],             "prg.copulas.archimedean.gumbel"
    FGM      = 5,  "FGM",     "Farlie-Gumbel-Morgenstern", "CopulaFGM", True,  ["tau_k"], [-2.0 / 9.0, 2.0 / 9.0],     "prg.copulas.explicit.fgm"
    CUBSEC   = 6,  "CubSec",  "Cubic Section",        "CopulaCubSec",   True,  ["tau_k"], [0.0, 33.0 / 200.0],          "prg.copulas.explicit.cubic_section"
    CLAYTON  = 7,  "Clayton", "Clayton",              "CopulaClayton",  True,  ["tau_k"], [0.0 + EPS, 1.0],             "prg.copulas.archimedean.clayton"
    A12      = 8,  "A12",     "Archimedean12",        "CopulaA12",      True,  ["tau_k"], [1.0 / 3.0, 1.0],             "prg.copulas.archimedean.a12"
    A14      = 9,  "A14",     "Archimedean14",        "CopulaA14",      True,  ["tau_k"], [1.0 / 3.0, 1.0],             "prg.copulas.archimedean.a14"
    FRANK           = 10, "Frank",    "Frank",                    "CopulaFrank",    True,  ["tau_k"], [EPS_MINUS_ONE, ONE_MINUS_EPS],  "prg.copulas.archimedean.frank"
    JOE             = 11, "Joe",      "Joe",                      "CopulaJoe",      True,  ["tau_k"], [0.0 + EPS, 1.0],              "prg.copulas.archimedean.joe"
    SURVIVAL_CLAYTON = 12, "SClayton", "Survival Clayton",        "SurvivalClayton", True, ["tau_k"], [0.0 + EPS, 1.0],             "prg.copulas.archimedean.survival"
    SURVIVAL_GH      = 13, "SGH",      "Survival Gumbel-Hougaard", "SurvivalGH",    True,  ["tau_k"], [0.0 + EPS, 1.0],             "prg.copulas.archimedean.survival"
    SURVIVAL_JOE     = 14, "SJoe",     "Survival Joe",            "SurvivalJoe",    True,  ["tau_k"],          [0.0 + EPS, 1.0],  "prg.copulas.archimedean.survival"
    BB1              = 15, "BB1",      "BB1 (Joe-Clayton)",       "CopulaBB1",      True,  ["tau_k", "delta"], [0.0 + EPS, 1.0],              "prg.copulas.archimedean.bb1"
    AMH              = 16, "AMH",      "Ali-Mikhail-Haq",         "CopulaAMH",      True,  ["tau_k"],          [_AMH_TAU_MIN, _AMH_TAU_MAX],  "prg.copulas.archimedean.amh"
    PLACKETT         = 17, "Plackett", "Plackett",                "CopulaPlackett", True,  ["tau_k"],          [EPS_MINUS_ONE, ONE_MINUS_EPS], "prg.copulas.explicit.plackett"

    def describe(self):
        return self.name, self.value

    def correct_tau(self, tau: float) -> float:
        """Clip tau to the valid range; replace non-finite values with the midpoint."""
        if not np.isfinite(tau):
            return (self.value.TAU_MIN_MAX[0] + self.value.TAU_MIN_MAX[1]) / 2.0
        return float(np.clip(tau, self.value.TAU_MIN_MAX[0], self.value.TAU_MIN_MAX[1]))

    @classmethod
    def favorite(cls):
        return cls.GAUSSIAN

    @classmethod
    def print_available(cls):
        logger.info("Available copulas:")
        for c in CopulaEnum:
            if c.value.AVAILABLE:
                logger.info("  %s", c.describe())

    @classmethod
    def available(cls):
        return [c for c in CopulaEnum if c.value.AVAILABLE]

    @classmethod
    def available_class_names(cls):
        return [c.CLASS_NAME for c in CopulaEnum if c.value.AVAILABLE]

    @classmethod
    def from_short_name(cls, short: str):
        for c in CopulaEnum:
            if c.value.SHORT_NAME == short:
                return c
        return None


class CopulaVirt:

    # Number of free parameters; subclasses override if they fit more than τ_K
    n_params: int = 1

    def __init__(self, class_name: str, params: dict):
        self.class_name = class_name

        self.copula_enum = None
        for c in CopulaEnum.available():
            if c.CLASS_NAME == self.class_name:
                self.copula_enum = c
                self.tau_min = c.value.TAU_MIN_MAX[0]
                self.tau_max = c.value.TAU_MIN_MAX[1]
                break
        if self.copula_enum is None:
            raise CopulaNotAvailableError(
                f"Copula {self.class_name!r} is not available."
            )

        for k in params:
            if k not in self.copula_enum.PARAMETERS_SET_NAME:
                raise CopulaParameterError(
                    f"Unexpected parameter {k!r} for {self.class_name}. "
                    f"Expected: {self.copula_enum.PARAMETERS_SET_NAME}"
                )

        if "tau_k" not in params:
            raise CopulaParameterError(
                f"Missing required parameter tau_k for {self.class_name}."
            )

        if not (self.tau_min <= params["tau_k"] <= self.tau_max):
            raise CopulaParameterError(
                f'tau_k={params["tau_k"]} out of [{self.tau_min}, {self.tau_max}] '
                f"for {self.class_name}."
            )

        self.params = params

        self._update_params()

        # Grid for the numerical majorant (150 points on (0,1))
        self.N = 150
        self.ticks_nbr = 15
        self._x = np.linspace(EPS, ONE_MINUS_EPS, self.N)
        self._y = np.zeros(self.N)

        # Grid for plotting (150×150)
        self._x1 = np.linspace(EPS, ONE_MINUS_EPS, self.N)
        self._y1 = np.linspace(EPS, ONE_MINUS_EPS, self.N)
        self._x2, self._y2 = np.meshgrid(self._x1, self._y1)
        self._z2 = np.zeros(self._x2.shape)

    # ------------------------------------------------------------------
    # Internal parameter update (must be overridden by every subclass)
    # ------------------------------------------------------------------
    def _update_params(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _update_params()."
        )

    # ------------------------------------------------------------------
    # PDF / CDF (must be overridden by every subclass)
    # ------------------------------------------------------------------
    def pdf(self, uv):
        raise NotImplementedError(f"{self.__class__.__name__} must implement pdf().")

    def cdf(self, uv):
        raise NotImplementedError(f"{self.__class__.__name__} must implement cdf().")

    # ------------------------------------------------------------------
    # Conditional CDF  h(v|u) = ∂C(u,v)/∂u
    # ------------------------------------------------------------------
    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = ∂C(u,v)/∂u via central finite differences. Subclasses override analytically."""
        h = 1e-5
        u_lo = minmaxEPS(u - h)
        u_hi = minmaxEPS(u + h)
        result = (self.cdf([u_hi, v]) - self.cdf([u_lo, v])) / (u_hi - u_lo)
        return float(np.clip(result, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Tail dependence coefficients  (λ_L, λ_U)
    # ------------------------------------------------------------------
    def tail_dependence(self) -> tuple[float, float]:
        """Lower/upper tail dependence (λ_L, λ_U).

        Default numerical implementation via diagonal CDF limits:
            λ_L = lim_{u→0+} C(u,u)/u
            λ_U = lim_{u→1−} (1 − 2u + C(u,u))/(1 − u)
        Subclasses with analytical formulas override this.
        Returns (nan, nan) if the copula has no closed-form CDF.
        """
        try:
            u0, u1 = 1e-4, 1.0 - 1e-4
            c0 = float(self.cdf([u0, u0]))
            c1 = float(self.cdf([u1, u1]))
            lam_L = c0 / u0
            lam_U = (1.0 - 2.0 * u1 + c1) / (1.0 - u1)
            return (float(np.clip(lam_L, 0.0, 1.0)),
                    float(np.clip(lam_U, 0.0, 1.0)))
        except NotImplementedError:
            return float('nan'), float('nan')

    # ------------------------------------------------------------------
    # Majorant  max_{v} c(u_left, v)  — overridden analytically when possible
    # ------------------------------------------------------------------
    def majorant(self, u_left: float) -> float:
        """Numerical majorant via grid search on 150 points. Subclasses may override."""
        u = minmaxEPS(u_left)
        for i, v in enumerate(self._x):
            self._y[i] = self.pdf([u, v])
        return float(np.max(self._y))

    # ------------------------------------------------------------------
    # tau update
    # ------------------------------------------------------------------
    def update_tau_k(self, new_tau_k: float):
        if self.tau_min <= new_tau_k <= self.tau_max:
            self.params["tau_k"] = new_tau_k
            self._update_params()
        else:
            self.params["tau_k"] = (self.tau_min + self.tau_max) / 2.0

    @property
    def tau_range(self) -> tuple[float, float]:
        return self.tau_min, self.tau_max

    # ------------------------------------------------------------------
    # Parameter estimation
    # ------------------------------------------------------------------
    @classmethod
    def fit(cls, data: np.ndarray, method: str = 'tau') -> 'FitResult':
        """Fit tau_k from a (n, 2) data array.

        Raw data is rank-transformed to pseudo-observations û = rank/(n+1)
        before estimation — no distributional assumption on the margins.

        Parameters
        ----------
        data   : array-like, shape (n, 2)
        method : {'tau', 'mle'}
            'tau' — moment matching via Kendall's τ (O(n log n), default)
            'mle' — maximize ∑ log c(û_i, v̂_i) via Brent scalar search

        Returns
        -------
        FitResult
        """
        from scipy.stats    import rankdata, kendalltau as _kendalltau
        from scipy.optimize import minimize_scalar

        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError(f'data must be shape (n, 2), got {data.shape}.')
        n = data.shape[0]
        if n < 4:
            raise ValueError(f'At least 4 observations required, got {n}.')

        uv = np.column_stack([
            rankdata(data[:, 0]) / (n + 1),
            rankdata(data[:, 1]) / (n + 1),
        ])

        class_name = cls.__name__
        tau_min = tau_max = None
        for entry in CopulaEnum:
            if entry.CLASS_NAME == class_name:
                if not entry.value.AVAILABLE:
                    raise CopulaNotAvailableError(
                        f'{class_name} is currently disabled.')
                tau_min, tau_max = entry.value.TAU_MIN_MAX
                break
        if tau_min is None:
            raise CopulaNotAvailableError(
                f'{class_name!r} is not registered in CopulaEnum.')

        # Single-point family (Product copula)
        if abs(tau_max - tau_min) < 1e-8:
            tau_k  = tau_min
            copula = cls(tau_k=tau_k)
            return FitResult(copula=copula, method=method, tau_k=tau_k,
                             log_likelihood=_eval_log_likelihood(copula, uv),
                             n_obs=n, uv=uv)

        if method == 'tau':
            tau_hat, _ = _kendalltau(data[:, 0], data[:, 1])
            tau_k = float(np.clip(tau_hat, tau_min, tau_max))

        elif method == 'mle':
            pad = max(1e-4 * (tau_max - tau_min), 1e-9)
            lo, hi = tau_min + pad, tau_max - pad

            def _neg_ll(tau: float) -> float:
                try:
                    return -_eval_log_likelihood(cls(tau_k=float(tau)), uv)
                except Exception:
                    return 1e12

            res   = minimize_scalar(_neg_ll, bounds=(lo, hi), method='bounded')
            tau_k = float(res.x)

        else:
            raise ValueError(f"method must be 'tau' or 'mle', got {method!r}.")

        copula = cls(tau_k=tau_k)
        return FitResult(copula=copula, method=method, tau_k=tau_k,
                         log_likelihood=_eval_log_likelihood(copula, uv),
                         n_obs=n, uv=uv)

    @staticmethod
    def fit_best(data: np.ndarray, families: list | None = None,
                 method: str = 'tau') -> list['FitResult']:
        """Fit each candidate family and return them sorted by AIC (smallest first).

        Parameters
        ----------
        data     : array-like, shape (n, 2)
        families : list of CopulaVirt subclasses, or None for a sensible default
                   (all available 1-parameter families except Product).
        method   : passed to each cls.fit()

        Returns
        -------
        list[FitResult]  sorted by AIC ascending. Failures are logged and skipped.
        """
        if families is None:
            import importlib as _il
            families = []
            for _entry in CopulaEnum:
                if not _entry.value.AVAILABLE or not _entry.MODULE:
                    continue
                if _entry.CLASS_NAME == 'CopulaProduct':   # τ fixé à 0, pas de fit utile
                    continue
                _mod = _il.import_module(_entry.MODULE)
                families.append(getattr(_mod, _entry.CLASS_NAME))

        results: list = []
        for cls in families:
            try:
                results.append(cls.fit(data, method=method))
            except Exception as e:
                logger.warning('%s.fit failed: %s', cls.__name__, e)
        results.sort(key=lambda r: r.aic)
        return results

    # ------------------------------------------------------------------
    # Representations
    # ------------------------------------------------------------------
    def __repr__(self):
        return str(self)

    def __str__(self):
        return (
            f"Copula name = {self.copula_enum}; Tau Kendall= {self.params['tau_k']:.2f}"
        )

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------
    def plot_pdf(self, plot_dir: str, prefix: str = "") -> None:
        """Contour plot of the copula density c(u,v) on (0,1)²."""
        Z = np.vectorize(lambda a, b: self.pdf([a, b]))(self._x2, self._y2)
        fig, ax = plt.subplots(figsize=(5, 5), facecolor=facecolor)
        vmax = np.nanpercentile(Z, 97)
        vticks = np.linspace(0, max(vmax, 1e-10), self.ticks_nbr)
        cs = ax.contourf(
            self._x2, self._y2, Z, vticks, cmap="viridis", vmin=0, vmax=vmax
        )
        fig.colorbar(cs, ax=ax, ticks=vticks[::3])
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_aspect("equal")
        theta_str = f", θ={self.theta:.3f}" if hasattr(self, "theta") else ""
        fig.suptitle(
            f"{self.copula_enum.LONG_NAME}  c(u,v)  τ={self.params['tau_k']}{theta_str}",
            y=0.98,
            fontsize=BIGGER_SIZE,
        )
        plt.savefig(
            os.path.join(
                plot_dir,
                f"{prefix}PDF_{self.copula_enum.SHORT_NAME}_tau{self.params['tau_k']:.2f}.png",
            ),
            bbox_inches="tight",
            dpi=dpi,
            facecolor=facecolor,
        )
        plt.close()

    def plot_cdf(self, plot_dir: str, prefix: str = "") -> None:
        """Contour plot of the copula CDF C(u,v) on (0,1)²."""
        try:
            Z = np.vectorize(lambda a, b: self.cdf([a, b]))(self._x2, self._y2)
        except NotImplementedError as e:
            logger.warning("plot_cdf skipped: %s", e)
            return
        fig, ax = plt.subplots(figsize=(5, 5), facecolor=facecolor)
        vticks = np.linspace(0, 1, self.ticks_nbr)
        cs = ax.contourf(self._x2, self._y2, Z, vticks, cmap="viridis", vmin=0, vmax=1)
        fig.colorbar(cs, ax=ax, ticks=vticks[::3])
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_aspect("equal")
        theta_str = f", θ={self.theta:.3f}" if hasattr(self, "theta") else ""
        fig.suptitle(
            f"{self.copula_enum.LONG_NAME}  C(u,v)  τ={self.params['tau_k']}{theta_str}",
            y=0.98,
            fontsize=BIGGER_SIZE,
        )
        plt.savefig(
            os.path.join(
                plot_dir,
                f"{prefix}CDF_{self.copula_enum.SHORT_NAME}_tau{self.params['tau_k']:.2f}.png",
            ),
            bbox_inches="tight",
            dpi=dpi,
            facecolor=facecolor,
        )
        plt.close()

    # ------------------------------------------------------------------
    # Sampling via Rosenblatt / h-function inversion
    # ------------------------------------------------------------------

    def sample(self, n: int = 500, seed: int | None = None) -> np.ndarray:
        """Draw n samples from the copula on [0,1]² via conditional inversion.

        Algorithm (Rosenblatt transform):
          u ~ Uniform(0,1)
          v = h⁻¹(w | u)  where  w ~ Uniform(0,1) and  h = conditional_cdf

        Returns array of shape (n, 2).
        """
        from scipy.optimize import brentq

        rng = np.random.default_rng(seed)
        us = rng.uniform(EPS, ONE_MINUS_EPS, n)
        ws = rng.uniform(EPS, ONE_MINUS_EPS, n)
        out = np.empty((n, 2))
        for i in range(n):
            u, w = float(us[i]), float(ws[i])
            out[i, 0] = u
            h_lo = self.conditional_cdf(EPS, u)
            h_hi = self.conditional_cdf(ONE_MINUS_EPS, u)
            if w <= h_lo:
                out[i, 1] = EPS
            elif w >= h_hi:
                out[i, 1] = ONE_MINUS_EPS
            else:
                out[i, 1] = brentq(
                    lambda v_: self.conditional_cdf(float(v_), u) - w,
                    EPS,
                    ONE_MINUS_EPS,
                    maxiter=60,
                    xtol=1e-6,
                )
        return out

    # ------------------------------------------------------------------
    # New plots
    # ------------------------------------------------------------------

    def plot_samples(
        self, plot_dir: str, prefix: str = "", n: int = 5000, seed: int = 42
    ) -> None:
        """Scatter of n copula samples overlaid on the PDF background."""
        samples = self.sample(n, seed=seed)
        Z = np.vectorize(lambda a, b: self.pdf([a, b]))(self._x2, self._y2)
        vmax = np.nanpercentile(Z, 97)

        fig, ax = plt.subplots(figsize=(5, 5), facecolor=facecolor)
        ticks = np.linspace(0, max(vmax, 1e-10), 8)
        ax.contourf(
            self._x2, self._y2, Z, ticks, cmap="Blues", alpha=0.45, vmin=0, vmax=vmax
        )
        ax.scatter(
            samples[:, 0], samples[:, 1], s=4, alpha=0.5, color="navy", linewidths=0
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_aspect("equal")
        theta_str = f", θ={self.theta:.3f}" if hasattr(self, "theta") else ""
        fig.suptitle(
            f"{self.copula_enum.LONG_NAME}  τ={self.params['tau_k']}{theta_str}  (n={n})",
            y=0.98,
            fontsize=BIGGER_SIZE,
        )
        plt.savefig(
            os.path.join(
                plot_dir,
                f"{prefix}Samples_{self.copula_enum.SHORT_NAME}_tau{self.params['tau_k']:.2f}.png",
            ),
            bbox_inches="tight",
            dpi=dpi,
            facecolor=facecolor,
        )
        plt.close()

    def plot_h_function(self, plot_dir: str, prefix: str = "") -> None:
        """Heatmap of h(v|u) = ∂C(u,v)/∂u with iso-probability contours.

        For the independence copula h(v|u) = v → horizontal stripes.
        Iso-contours (dashed) show levels [0.1, 0.25, 0.5, 0.75, 0.9].
        """
        H = np.vectorize(lambda a, b: self.conditional_cdf(float(b), float(a)))(
            self._x2, self._y2
        )

        fig, ax = plt.subplots(figsize=(5, 5), facecolor=facecolor)
        im = ax.pcolormesh(
            self._x2, self._y2, H, cmap="viridis", vmin=0, vmax=1, shading="auto"
        )
        fig.colorbar(im, ax=ax, label="h(v|u)")
        ax.contour(
            self._x2,
            self._y2,
            H,
            levels=[0.1, 0.25, 0.5, 0.75, 0.9],
            colors="white",
            linewidths=0.6,
            alpha=0.7,
            linestyles="--",
        )
        ax.set_xlabel("u  (conditioning)")
        ax.set_ylabel("v")
        ax.set_aspect("equal")
        theta_str = f", θ={self.theta:.3f}" if hasattr(self, "theta") else ""
        fig.suptitle(
            f"h(v|u) = ∂C/∂u  —  {self.copula_enum.LONG_NAME}  τ={self.params['tau_k']}{theta_str}",
            y=0.98,
            fontsize=BIGGER_SIZE,
        )
        plt.savefig(
            os.path.join(
                plot_dir,
                f"{prefix}Hfunc_{self.copula_enum.SHORT_NAME}_tau{self.params['tau_k']:.2f}.png",
            ),
            bbox_inches="tight",
            dpi=dpi,
            facecolor=facecolor,
        )
        plt.close()

    def plot_overview(
        self, plot_dir: str, prefix: str = "", n_samples: int = 2000, seed: int = 42
    ) -> None:
        """2×2 panel: PDF · CDF · h-function · samples."""
        tau = self.params["tau_k"]
        name = self.copula_enum.value.LONG_NAME
        theta_str = f", θ={self.theta:.3f}" if hasattr(self, "theta") else ""

        Z_pdf = np.vectorize(lambda a, b: self.pdf([a, b]))(self._x2, self._y2)
        Z_h = np.vectorize(lambda a, b: self.conditional_cdf(float(b), float(a)))(
            self._x2, self._y2
        )
        try:
            Z_cdf = np.vectorize(lambda a, b: self.cdf([a, b]))(self._x2, self._y2)
            has_cdf = True
        except NotImplementedError:
            has_cdf = False
        samples = self.sample(n_samples, seed=seed)

        fig, axes = plt.subplots(2, 2, figsize=(10, 10), facecolor=facecolor)
        fig.suptitle(f"{name}  (τ={tau}{theta_str})", fontsize=BIGGER_SIZE + 2)
        fig.subplots_adjust(top=0.93, wspace=0.32, hspace=0.32)

        # PDF
        ax = axes[0, 0]
        vmax = np.nanpercentile(Z_pdf, 97)
        tks = np.linspace(0, max(vmax, 1e-10), self.ticks_nbr)
        cs = ax.contourf(
            self._x2, self._y2, Z_pdf, tks, cmap="viridis", vmin=0, vmax=vmax
        )
        fig.colorbar(cs, ax=ax, ticks=tks[::3])
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_aspect("equal")
        ax.set_title("PDF  c(u, v)", fontsize=BIGGER_SIZE)

        # CDF
        ax = axes[0, 1]
        if has_cdf:
            tks = np.linspace(0, 1, self.ticks_nbr)
            cs = ax.contourf(
                self._x2, self._y2, Z_cdf, tks, cmap="viridis", vmin=0, vmax=1
            )
            fig.colorbar(cs, ax=ax, ticks=tks[::3])
            ax.set_xlabel("u")
            ax.set_ylabel("v")
            ax.set_aspect("equal")
        else:
            ax.text(
                0.5,
                0.5,
                "CDF not available\nin closed form",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        ax.set_title("CDF  C(u, v)", fontsize=BIGGER_SIZE)

        # h-function
        ax = axes[1, 0]
        im = ax.pcolormesh(
            self._x2, self._y2, Z_h, cmap="viridis", vmin=0, vmax=1, shading="auto"
        )
        fig.colorbar(im, ax=ax)
        ax.contour(
            self._x2,
            self._y2,
            Z_h,
            levels=[0.1, 0.25, 0.5, 0.75, 0.9],
            colors="white",
            linewidths=0.5,
            alpha=0.6,
            linestyles="--",
        )
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_aspect("equal")
        ax.set_title("h(v|u) = ∂C/∂u", fontsize=BIGGER_SIZE)

        # Samples
        ax = axes[1, 1]
        tks = np.linspace(0, max(vmax, 1e-10), 8)
        ax.contourf(
            self._x2, self._y2, Z_pdf, tks, cmap="Blues", alpha=0.4, vmin=0, vmax=vmax
        )
        ax.scatter(
            samples[:, 0], samples[:, 1], s=3, alpha=0.5, color="navy", linewidths=0
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_aspect("equal")
        ax.set_title(f"Samples  (n={n_samples})", fontsize=BIGGER_SIZE)

        plt.savefig(
            os.path.join(
                plot_dir,
                f"{prefix}Overview_{self.copula_enum.SHORT_NAME}_tau{tau:.2f}.png",
            ),
            bbox_inches="tight",
            dpi=dpi,
            facecolor=facecolor,
        )
        plt.close()

    def plot_multi_tau(
        self,
        plot_dir: str,
        prefix: str = "",
        tau_values: list | None = None,
        ncols: int = 3,
    ) -> None:
        """Grid of PDF contours for several τ values — illustrates the copula family."""
        tau_min, tau_max = self.tau_range
        if abs(tau_max - tau_min) < 1e-8:
            return
        if tau_values is None:
            span = tau_max - tau_min
            pad = min(0.1 * span, 0.05)
            tau_values = list(np.linspace(tau_min + pad, tau_max - pad, ncols * 2))

        n = len(tau_values)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4 * ncols, 4 * nrows),
            facecolor=facecolor,
            squeeze=False,
        )
        fig.suptitle(
            f"{self.copula_enum.value.LONG_NAME}  —  PDF c(u,v) for various τ",
            fontsize=BIGGER_SIZE + 2,
            y=1.01,
        )
        for idx, tau in enumerate(tau_values):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]
            cop = self.__class__(**{**self.params, 'tau_k': float(tau)})
            Z = np.vectorize(lambda a, b: cop.pdf([a, b]))(cop._x2, cop._y2)
            vmax = np.nanpercentile(Z, 97)
            tks = np.linspace(0, max(vmax, 1e-10), 9)
            ax.contourf(cop._x2, cop._y2, Z, tks, cmap="viridis", vmin=0, vmax=vmax)
            theta_str = f", θ={cop.theta:.2f}" if hasattr(cop, "theta") else ""
            ax.set_title(f"τ = {tau:.2f}{theta_str}", fontsize=BIGGER_SIZE)
            ax.set_xlabel("u")
            ax.set_ylabel("v")
            ax.set_aspect("equal")

        for idx in range(n, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                plot_dir, f"{prefix}MultiTau_{self.copula_enum.SHORT_NAME}.png"
            ),
            bbox_inches="tight",
            dpi=dpi,
            facecolor=facecolor,
        )
        plt.close()


# ---------------------------------------------------------------------------
# Fitting helpers (module-level so FitResult is importable)
# ---------------------------------------------------------------------------

def _eval_log_likelihood(copula: CopulaVirt, uv: np.ndarray) -> float:
    """∑ log c(û_i, v̂_i) on pseudo-observations, flooring pdf at EPS."""
    pdf_vals = np.array([copula.pdf(list(row)) for row in uv])
    return float(np.sum(np.log(np.maximum(pdf_vals, EPS))))


def _empirical_copula(uv: np.ndarray, points: np.ndarray) -> np.ndarray:
    """C_n(u,v) = (1/n) Σ 1{û_i ≤ u, v̂_i ≤ v}, evaluated at given points."""
    leq_u = uv[:, 0][None, :] <= points[:, 0][:, None]   # (m, n)
    leq_v = uv[:, 1][None, :] <= points[:, 1][:, None]
    return (leq_u & leq_v).mean(axis=1)


def _cvm_statistic(uv: np.ndarray, copula: CopulaVirt) -> float:
    """Cramér-von Mises statistic S_n = Σ (C_n(u_i,v_i) − C_θ(u_i,v_i))²."""
    Cn     = _empirical_copula(uv, uv)
    Ctheta = np.array([copula.cdf(list(row)) for row in uv])
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


@dataclass
class FitResult:
    """Returned by :meth:`CopulaVirt.fit`."""
    copula:         CopulaVirt
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
        for b in range(B):
            sample = self.copula.sample(n=self.n_obs,
                                        seed=int(rng.integers(0, 2**31 - 1)))
            try:
                r_b      = cls.fit(sample, method=self.method)
                stats[b] = _cvm_statistic(r_b.uv, r_b.copula)
            except Exception as e:
                logger.debug('Bootstrap iter %d failed: %s', b, e)

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
        for b in range(B):
            idx    = rng.integers(0, self.n_obs, size=self.n_obs)
            sample = self.uv[idx]
            try:
                tau_boots[b] = cls.fit(sample, method=self.method).tau_k
            except Exception as e:
                logger.debug('Bootstrap iter %d failed: %s', b, e)
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
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor=facecolor)
        fig.suptitle(
            f'{name} — fit diagnostics  (τ̂={self.tau_k:.3f}, n={self.n_obs})',
            fontsize=BIGGER_SIZE + 1, y=0.995,
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
        ax.set_title('Pseudo-obs + fitted PDF', fontsize=BIGGER_SIZE)
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
            ax.set_title('PP plot', fontsize=BIGGER_SIZE)

            # (1,0) — empirical copula
            ax = axes[1, 0]
            cs = ax.contourf(GX, GY, Cn_g, levels=11, cmap='viridis', vmin=0, vmax=1)
            fig.colorbar(cs, ax=ax)
            ax.set_title(r'$C_n(u,v)$ (empirical)', fontsize=BIGGER_SIZE)
            ax.set_xlabel('u'); ax.set_ylabel('v'); ax.set_aspect('equal')

            # (1,1) — residuals
            ax    = axes[1, 1]
            diff  = Cn_g - Ct_g
            vmaxd = float(max(abs(diff.min()), abs(diff.max()), 1e-3))
            cs = ax.contourf(GX, GY, diff, levels=15, cmap='RdBu_r',
                             vmin=-vmaxd, vmax=vmaxd)
            fig.colorbar(cs, ax=ax)
            ax.set_title(rf'Residuals $C_n - C_\theta$  (max |Δ|={vmaxd:.3f})',
                         fontsize=BIGGER_SIZE)
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
        ax.set_title('Lower tail dependence', fontsize=BIGGER_SIZE)
        ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=9)

        # (1,2) — upper tail dependence
        ax = axes[1, 2]
        ax.plot(u_high, lU_emp, 'o-', ms=4, label=r'Empirical $\hat\lambda_U(u)$')
        if not np.isnan(lU_th):
            ax.axhline(lU_th, color='red', ls='--', lw=1.2,
                       label=rf'Fitted $\lambda_U = {lU_th:.3f}$')
        ax.set_xlabel('u'); ax.set_ylabel(r'$\lambda_U$')
        ax.set_title('Upper tail dependence', fontsize=BIGGER_SIZE)
        ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{prefix}Diagnostics_{short}.png'),
                    bbox_inches='tight', dpi=dpi, facecolor=facecolor)
        plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    for c in CopulaEnum.available():
        print(c.describe())
    print("favorite:", CopulaEnum.favorite())
