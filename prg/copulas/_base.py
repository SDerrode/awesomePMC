if __name__ == "__main__":
    import sys
    import pathlib

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import logging
import os
import matplotlib.pyplot as plt
import numpy as np

from enum import Enum, unique
from dataclasses import dataclass, field

from prg.numerics import EPS, ONE_MINUS_EPS, EPS_MINUS_ONE, minmaxEPS
from prg.plot_style import DEFAULT_FONT_SIZE as FONT_SIZE
from prg.exceptions import CopulaParameterError, CopulaNotAvailableError

# AMH τ_K range constants (computed at import time, θ ∈ [−1, 1))
_AMH_TAU_MIN = float(1.0 - 2.0 * (4.0 * np.log(2.0) - 1.0) / 3.0)  # τ(θ=−1) ≈ −0.1817
_AMH_TAU_MAX = float(1.0 / 3.0 - EPS)                                  # limit τ → 1/3

logger = logging.getLogger(__name__)

# Module-level cache for CopulaEnum.klass (lazy import).
_COPULA_KLASS_CACHE: dict = {}


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
    STUDENT  = 3,  "Student", "Student",              "CopulaStudent",  True,  ["tau_k", "df"], [-1.0, 1.0],             "prg.copulas.elliptical.student"
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
        """Clip tau to the valid range; replace non-finite values with the midpoint.

        Logs a ``WARNING`` when the input is outside the family's valid τ
        range or non-finite, so that callers (e.g. ICE candidate selection)
        notice the silent clipping.
        """
        tau_min, tau_max = self.value.TAU_MIN_MAX
        if not np.isfinite(tau):
            mid = 0.5 * (tau_min + tau_max)
            logger.warning(
                "%s.correct_tau: τ=%r is non-finite — using midpoint τ=%.4f.",
                self.value.SHORT_NAME, tau, mid,
            )
            return mid
        if tau < tau_min or tau > tau_max:
            clipped = float(np.clip(tau, tau_min, tau_max))
            logger.warning(
                "%s.correct_tau: τ=%.4f outside valid range [%.4f, %.4f] — "
                "clipped to τ=%.4f.",
                self.value.SHORT_NAME, tau, tau_min, tau_max, clipped,
            )
            return clipped
        return float(tau)

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
        return [c.value.CLASS_NAME for c in CopulaEnum if c.value.AVAILABLE]

    @classmethod
    def from_short_name(cls, short: str):
        for c in CopulaEnum:
            if c.value.SHORT_NAME == short:
                return c
        return None

    @property
    def klass(self):
        """The implementation class for this copula, imported lazily and cached.

        Avoids re-running ``importlib.import_module`` on every model build /
        ICE candidate evaluation. Keyed by the (immutable) ``CLASS_NAME``
        because ``CopulaEnum`` instances are not hashable (the dataclass
        mixin overrides ``__eq__``).
        """
        key = self.value.CLASS_NAME
        cls_obj = _COPULA_KLASS_CACHE.get(key)
        if cls_obj is None:
            import importlib as _importlib
            mod = _importlib.import_module(self.value.MODULE)
            cls_obj = getattr(mod, self.value.CLASS_NAME)
            _COPULA_KLASS_CACHE[key] = cls_obj
        return cls_obj


class CopulaVirt:

    # Number of free parameters; subclasses override if they fit more than τ_K
    n_params: int = 1

    def __init__(self, class_name: str, params: dict):
        self.class_name = class_name

        self.copula_enum = None
        for c in CopulaEnum.available():
            if c.value.CLASS_NAME == self.class_name:
                self.copula_enum = c
                self.tau_min = c.value.TAU_MIN_MAX[0]
                self.tau_max = c.value.TAU_MIN_MAX[1]
                break
        if self.copula_enum is None:
            raise CopulaNotAvailableError(
                f"Copula {self.class_name!r} is not available."
            )

        for k in params:
            if k not in self.copula_enum.value.PARAMETERS_SET_NAME:
                raise CopulaParameterError(
                    f"Unexpected parameter {k!r} for {self.class_name}. "
                    f"Expected: {self.copula_enum.value.PARAMETERS_SET_NAME}"
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
    # Vectorised PDF — used by hot loops in prg.pmc (forward-backward, ICE).
    #
    # Default fast path: most subclasses wrap a `statsmodels` copula in
    # ``self._model``, whose ``.pdf()`` accepts a (M, 2) ndarray and returns
    # an (M,) ndarray; we exploit that. Subclasses without ``self._model``
    # (closed-form-only families like Joe) override this with a vectorised
    # closed form. As a last resort we fall back to a scalar Python loop.
    # ------------------------------------------------------------------
    def pdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Evaluate c(u, v) on M point pairs.

        Parameters
        ----------
        uv : np.ndarray, shape (M, 2)

        Returns
        -------
        np.ndarray, shape (M,) — copula PDF values, clipped to ≥ EPS.
        """
        uv = np.asarray(uv, dtype=float)
        if uv.ndim != 2 or uv.shape[1] != 2:
            raise ValueError(f"pdf_array expects shape (M, 2); got {uv.shape}.")

        # Clamp inputs to (EPS, 1-EPS) to mirror scalar minmaxEPS
        uv = np.clip(uv, EPS, ONE_MINUS_EPS)

        if hasattr(self, "_model") and hasattr(self._model, "pdf"):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                vals = np.asarray(self._model.pdf(uv), dtype=float)
            # Replace non-finite or non-positive values with EPS (mirrors scalar guards)
            vals = np.where(np.isfinite(vals) & (vals > 0.0), vals, EPS)
            return vals

        # Fallback — scalar loop (slow). Subclasses without ``_model`` should
        # override pdf_array with a vectorised closed form.
        out = np.empty(uv.shape[0], dtype=float)
        for k in range(uv.shape[0]):
            out[k] = self.pdf([float(uv[k, 0]), float(uv[k, 1])])
        return out

    def logpdf_array(self, uv: np.ndarray) -> np.ndarray:
        """log c(u, v) on M point pairs (default: log(pdf_array), floored at log(EPS))."""
        return np.log(np.maximum(self.pdf_array(uv), EPS))

    def cdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Evaluate C(u, v) on M point pairs.

        Same fast-path strategy as :meth:`pdf_array`: use the statsmodels
        backend if present (its ``.cdf`` accepts a (M, 2) ndarray and returns
        an (M,) ndarray), otherwise fall back to a Python loop.

        Subclasses with closed-form CDFs (FGM, AMH, Joe, …) can override this
        with a vectorised expression for ~50× speed-ups on bootstrap GoF.

        Parameters
        ----------
        uv : np.ndarray, shape (M, 2)

        Returns
        -------
        np.ndarray, shape (M,) — clipped to [0, 1].
        """
        uv = np.asarray(uv, dtype=float)
        if uv.ndim != 2 or uv.shape[1] != 2:
            raise ValueError(f"cdf_array expects shape (M, 2); got {uv.shape}.")

        uv = np.clip(uv, EPS, ONE_MINUS_EPS)

        if hasattr(self, "_model") and hasattr(self._model, "cdf"):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                vals = np.asarray(self._model.cdf(uv), dtype=float)
            vals = np.where(np.isfinite(vals), vals, EPS)
            return np.clip(vals, 0.0, 1.0)

        # Fallback — scalar loop (slow). Subclasses without ``_model`` should
        # override cdf_array with a vectorised closed form.
        out = np.empty(uv.shape[0], dtype=float)
        for k in range(uv.shape[0]):
            out[k] = self.cdf([float(uv[k, 0]), float(uv[k, 1])])
        return np.clip(out, 0.0, 1.0)

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
    # Inverse of the conditional CDF (Rosenblatt sampling step).
    #
    # Default: numerical inversion via Brent's method on h(v|u) = w. Subclasses
    # with an analytical form (e.g. Gaussian, Clayton) override this for a
    # ~50× speed-up on long simulations.
    # ------------------------------------------------------------------
    def inv_h(self, w: float, u: float) -> float:
        """Return v such that h(v|u) = w (Rosenblatt inverse step).

        Default numerical implementation: bracket ``h(v|u) - w`` on
        ``(EPS, 1-EPS)`` and solve via Brent's method. Brent requires the
        function to be monotone in v on that bracket — this holds for any
        valid copula (h is a conditional CDF in v) but the assertion below
        guards against an ill-defined custom copula.

        Subclasses with closed-form inverses (Gaussian, Clayton, Frank, …)
        override this for ~50× speed-up.
        """
        from scipy.optimize import brentq
        u    = minmaxEPS(u)
        h_lo = self.conditional_cdf(EPS,           u)
        h_hi = self.conditional_cdf(ONE_MINUS_EPS, u)

        # h(v|u) is a CDF in v → must be non-decreasing on (0, 1).
        # A violation here indicates a buggy ``conditional_cdf`` override.
        if h_lo > h_hi + 1e-9:
            raise ValueError(
                f"{type(self).__name__}.conditional_cdf is not monotone in v "
                f"at u={u:.4g} (h(EPS|u)={h_lo:.4g} > h(1-EPS|u)={h_hi:.4g}). "
                f"Cannot invert h."
            )

        if w <= h_lo:
            return float(EPS)
        if w >= h_hi:
            return float(ONE_MINUS_EPS)
        return float(brentq(
            lambda v_: self.conditional_cdf(float(v_), u) - w,
            EPS, ONE_MINUS_EPS, maxiter=80, xtol=1e-8,
        ))

    def inv_h_array(self, w: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Vectorised version of :meth:`inv_h`.

        Default fallback: scalar Python loop. Subclasses with closed-form
        ``inv_h`` (Gaussian, Clayton, Frank) override this with a
        vectorised expression for ~10× speed-up on bulk sampling.

        Parameters
        ----------
        w : np.ndarray, shape (M,) — uniform variates.
        u : np.ndarray, shape (M,) — conditioning values.

        Returns
        -------
        np.ndarray, shape (M,) — v values such that h(v_i | u_i) = w_i.
        """
        w = np.asarray(w, dtype=float)
        u = np.asarray(u, dtype=float)
        if w.shape != u.shape:
            raise ValueError(f"inv_h_array: w {w.shape} and u {u.shape} must match.")
        out = np.empty_like(w)
        for k in range(w.size):
            out[k] = self.inv_h(float(w[k]), float(u[k]))
        return out

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
        """Compute ``max_v c(u_left, v)`` for the AR-rejection sampler.

        Strategy (most-robust default):

        1. Coarse grid search (150 points) to localise the peak.
        2. Refinement via :func:`scipy.optimize.minimize_scalar` (Brent /
           bounded) in a tight neighbourhood of the grid maximum.
        3. **Safety multiplier ×1.02** to absorb residual numerical error.
           Acceptance rate drops by ≤2 % but the AR sampler is guaranteed
           unbiased.

        Subclasses with closed-form maxima (FGM, CubSec, Product, Gaussian,
        Clayton, …) override this method for ~10× speed-up and exact bounds.
        """
        from scipy.optimize import minimize_scalar
        u = minmaxEPS(u_left)

        # 1. Coarse grid search
        for i, v in enumerate(self._x):
            self._y[i] = self.pdf([u, v])
        idx_grid_max = int(np.argmax(self._y))
        v_grid_max   = self._x[idx_grid_max]
        f_grid_max   = self._y[idx_grid_max]

        # 2. Refinement around the grid max — bracket of width ~2 grid spacings
        spacing  = (ONE_MINUS_EPS - EPS) / max(self.N - 1, 1)
        v_lo     = float(max(EPS,         v_grid_max - 2.0 * spacing))
        v_hi     = float(min(ONE_MINUS_EPS, v_grid_max + 2.0 * spacing))
        try:
            res = minimize_scalar(
                lambda v_: -self.pdf([u, float(v_)]),
                bounds=(v_lo, v_hi),
                method="bounded",
                options={"xatol": 1e-7},
            )
            f_refined = self.pdf([u, float(res.x)])
            f_max     = max(f_grid_max, f_refined)
        except Exception:
            f_max = f_grid_max   # grid is the safety net

        # 3. Safety margin (constant 2 %) — guards against sub-grid peaks the
        #    refinement window may have missed.
        return float(f_max * 1.02)

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
        """Fit a copula's free parameters from a (n, 2) data array.

        Raw data is rank-transformed to pseudo-observations û = rank/(n+1)
        before estimation — no distributional assumption on the margins.

        Parameters
        ----------
        data   : array-like, shape (n, 2)
        method : {'tau', 'mle'}
            ``'tau'`` — moment matching via Kendall's τ (O(n log n), default).
            ``'mle'`` — maximise ∑ log c(û_i, v̂_i). For 1-parameter families
                       a 1-D Brent scalar search; for multi-parameter families
                       (Student, BB1) a 2-D L-BFGS-B optimisation jointly over
                       (τ_K, second parameter).

        Returns
        -------
        FitResult
        """
        from scipy.stats    import rankdata, kendalltau as _kendalltau
        from scipy.optimize import minimize, minimize_scalar

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
        param_names: list[str] = []
        for entry in CopulaEnum:
            if entry.value.CLASS_NAME == class_name:
                if not entry.value.AVAILABLE:
                    raise CopulaNotAvailableError(
                        f'{class_name} is currently disabled.')
                tau_min, tau_max = entry.value.TAU_MIN_MAX
                param_names = list(entry.value.PARAMETERS_SET_NAME)
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

        # Extra (non-tau) parameters and their bounds. Mirrors
        # ``prg.pmc.ice._EXTRA_PARAM_BOUNDS`` so that standalone fitting and
        # ICE-driven fitting use the same defaults.
        _EXTRA_BOUNDS = {
            "delta": (1.0,  10.0,  1.5),
            "df":    (2.0, 100.0,  4.0),
        }
        extras = {p: _EXTRA_BOUNDS[p] for p in param_names
                  if p != "tau_k" and p in _EXTRA_BOUNDS}

        if method == 'tau':
            tau_hat, _ = _kendalltau(data[:, 0], data[:, 1])
            tau_k = float(np.clip(tau_hat, tau_min, tau_max))
            extra_vals = {k: v[2] for k, v in extras.items()}   # init defaults

        elif method == 'mle':
            pad = max(1e-4 * (tau_max - tau_min), 1e-9)
            tau_lo, tau_hi = tau_min + pad, tau_max - pad

            if not extras:
                # 1-D scalar optimisation over τ
                def _neg_ll(tau: float) -> float:
                    try:
                        return -_eval_log_likelihood(cls(tau_k=float(tau)), uv)
                    except Exception:
                        return 1e12
                res   = minimize_scalar(_neg_ll, bounds=(tau_lo, tau_hi),
                                         method='bounded')
                tau_k = float(res.x)
                extra_vals = {}
            else:
                # n-D optimisation jointly over (τ, *extras) via L-BFGS-B
                bounds_list = [(tau_lo, tau_hi)]
                init        = [0.5 * (tau_min + tau_max)]
                ext_names: list[str] = []
                for name, (lo, hi, x0) in extras.items():
                    bounds_list.append((lo, hi))
                    init.append(x0)
                    ext_names.append(name)

                def _neg_ll_nd(theta):
                    kw = {"tau_k": float(theta[0])}
                    for k, name in enumerate(ext_names):
                        kw[name] = float(theta[k + 1])
                    try:
                        return -_eval_log_likelihood(cls(**kw), uv)
                    except Exception:
                        return 1e12

                try:
                    res = minimize(_neg_ll_nd, init,
                                   method="L-BFGS-B", bounds=bounds_list,
                                   options={"maxiter": 80, "ftol": 1e-7})
                    tau_k = float(np.clip(res.x[0], tau_lo, tau_hi))
                    extra_vals = {
                        name: float(np.clip(res.x[k + 1], *bounds_list[k + 1]))
                        for k, name in enumerate(ext_names)
                    }
                except Exception as exc:
                    logger.warning(
                        '%s.fit(method="mle"): joint optimisation failed (%s); '
                        'falling back to τ-only.', class_name, exc,
                    )
                    tau_k = init[0]
                    extra_vals = {k: v[2] for k, v in extras.items()}

        else:
            raise ValueError(f"method must be 'tau' or 'mle', got {method!r}.")

        copula = cls(tau_k=tau_k, **extra_vals)
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
                if not _entry.value.AVAILABLE or not _entry.value.MODULE:
                    continue
                if _entry.value.CLASS_NAME == 'CopulaProduct':   # τ fixé à 0, pas de fit utile
                    continue
                _mod = _il.import_module(_entry.value.MODULE)
                families.append(getattr(_mod, _entry.value.CLASS_NAME))

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
        fig, ax = plt.subplots(figsize=(5, 5))
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
        )
        plt.savefig(
            os.path.join(
                plot_dir,
                f"{prefix}PDF_{self.copula_enum.SHORT_NAME}_tau{self.params['tau_k']:.2f}.png",
            ),
            bbox_inches="tight",
        )
        plt.close()

    def plot_cdf(self, plot_dir: str, prefix: str = "") -> None:
        """Contour plot of the copula CDF C(u,v) on (0,1)²."""
        try:
            Z = np.vectorize(lambda a, b: self.cdf([a, b]))(self._x2, self._y2)
        except NotImplementedError as e:
            logger.warning("plot_cdf skipped: %s", e)
            return
        fig, ax = plt.subplots(figsize=(5, 5))
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
        )
        plt.savefig(
            os.path.join(
                plot_dir,
                f"{prefix}CDF_{self.copula_enum.SHORT_NAME}_tau{self.params['tau_k']:.2f}.png",
            ),
            bbox_inches="tight",
        )
        plt.close()

    # ------------------------------------------------------------------
    # Sampling via Rosenblatt / h-function inversion
    # ------------------------------------------------------------------

    def sample(self, n: int = 500, seed: int | None = None) -> np.ndarray:
        """Draw n samples from the copula on [0,1]² via Rosenblatt inversion.

        For each sample::

            u ~ Uniform(EPS, 1-EPS)
            w ~ Uniform(EPS, 1-EPS)
            v = inv_h(w, u)

        Subclasses with a closed-form ``inv_h_array`` (Gaussian, Clayton,
        Frank) get a single vectorised call for all n samples; others
        fall back to per-sample Brent inversion via ``inv_h``.
        """
        rng = np.random.default_rng(seed)
        us  = rng.uniform(EPS, ONE_MINUS_EPS, n)
        ws  = rng.uniform(EPS, ONE_MINUS_EPS, n)
        out = np.empty((n, 2))
        out[:, 0] = us
        out[:, 1] = self.inv_h_array(ws, us)
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

        fig, ax = plt.subplots(figsize=(5, 5))
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
        )
        plt.savefig(
            os.path.join(
                plot_dir,
                f"{prefix}Samples_{self.copula_enum.SHORT_NAME}_tau{self.params['tau_k']:.2f}.png",
            ),
            bbox_inches="tight",
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

        fig, ax = plt.subplots(figsize=(5, 5))
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
        )
        plt.savefig(
            os.path.join(
                plot_dir,
                f"{prefix}Hfunc_{self.copula_enum.SHORT_NAME}_tau{self.params['tau_k']:.2f}.png",
            ),
            bbox_inches="tight",
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

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle(f"{name}  (τ={tau}{theta_str})", fontsize=FONT_SIZE + 2)
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
        ax.set_title("PDF  c(u, v)")

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
        ax.set_title("CDF  C(u, v)")

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
        ax.set_title("h(v|u) = ∂C/∂u")

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
        ax.set_title(f"Samples  (n={n_samples})")

        plt.savefig(
            os.path.join(
                plot_dir,
                f"{prefix}Overview_{self.copula_enum.SHORT_NAME}_tau{tau:.2f}.png",
            ),
            bbox_inches="tight",
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
            squeeze=False,
        )
        fig.suptitle(
            f"{self.copula_enum.value.LONG_NAME}  —  PDF c(u,v) for various τ",
            fontsize=FONT_SIZE + 2,
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
            ax.set_title(f"τ = {tau:.2f}{theta_str}")
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
        )
        plt.close()




# ---------------------------------------------------------------------------
# Fitting helpers — re-exported from prg.copulas._fit for backward compatibility
# (consumers — bivariate.py, the public ``__init__``, and external code —
# import these symbols from ``_base``; the underscore-prefixed helpers are
# also used by the test suite).
# ---------------------------------------------------------------------------

from prg.copulas._fit import (   # noqa: E402, F401  (re-export at module bottom)
    _cvm_statistic,
    _empirical_copula,
    _empirical_tail_dep,
    _eval_log_likelihood,
    FitResult,
    GoFResult,
)
