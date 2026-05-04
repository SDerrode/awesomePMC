if __name__ == "__main__":
    import sys, pathlib

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import logging
import os
import secrets
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from matplotlib import gridspec

from prg.tools.tools import minmaxEPS, EPS
from prg.settings.plot_settings import facecolor, dpi, BIGGER_SIZE
from prg.exceptions import SamplingConvergenceError
from prg.copulas._base import FitResult, GoFResult, _empirical_tail_dep

logger = logging.getLogger(__name__)

_NBITERMAX = 80_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _margin_to_dict(margin: tuple) -> dict:
    """Convert (scipy_dist, *params) to an internal parameter dict."""
    dist = margin[0]
    return {
        "dist": dist,
        "dist_name": dist.name,
        "params": list(margin[1:]),
    }


def _validate_margin(margin: tuple, label: str) -> None:
    if not isinstance(margin, tuple) or len(margin) < 1:
        raise ValueError(f"{label} must be a non-empty tuple (scipy_dist, *params).")
    dist = margin[0]
    if not all(hasattr(dist, m) for m in ("pdf", "cdf", "logpdf", "rvs", "ppf")):
        raise ValueError(f"{label}[0] must be a scipy continuous distribution.")


# ---------------------------------------------------------------------------
# Frozen conditional law  p(Y_cond | Y_obs = y_obs)
# ---------------------------------------------------------------------------


class ConditionalLaw:
    """Frozen conditional distribution p(Y_cond | Y_obs = y_obs).

    Obtained via BivariateLaw.conditional_law(y_obs, which).
    All methods delegate to the parent BivariateLaw and share its RNG.

    Parameters
    ----------
    parent : BivariateLaw
    y_obs  : observed value of the conditioning dimension
    which  : 'left'  → condition on left  margin, output is right
             'right' → condition on right margin, output is left
    """

    def __init__(self, parent: "BivariateLaw", y_obs: float, which: str):
        if which not in ("left", "right"):
            raise ValueError("which must be 'left' or 'right'.")
        self._parent = parent
        self._y_obs = float(y_obs)
        self._which = which

    # --- scalar interface ---
    def pdf(self, y: float) -> float:
        """Conditional density p(y | y_obs)."""
        return self._parent.conditional_pdf(y, self._y_obs, which=self._which)

    def log_pdf(self, y: float) -> float:
        """Log conditional density."""
        return self._parent.conditional_log_pdf(y, self._y_obs, which=self._which)

    def cdf(self, y: float) -> float:
        """Conditional CDF P(Y_cond ≤ y | y_obs)."""
        return self._parent.conditional_cdf(y, self._y_obs, which=self._which)

    def sample(self, n: int = 1) -> np.ndarray:
        """Draw n iid samples. Returns shape (n,)."""
        return self._parent.sample_conditional(self._y_obs, which=self._which, n=n)

    def __repr__(self) -> str:
        return f"ConditionalLaw(which={self._which!r}, y_obs={self._y_obs:.4g})"


# ---------------------------------------------------------------------------
# Bivariate law (Sklar)
# ---------------------------------------------------------------------------


class BivariateLaw:
    """Bivariate distribution built from a copula and two marginals (Sklar's theorem):

        f(x, y) = f1(x) · f2(y) · c(F1(x), F2(y))

    Parameters
    ----------
    copula        : CopulaVirt instance
    left_margin   : tuple  (scipy_dist, *params)
    right_margin  : tuple  (scipy_dist, *params)
    quantile_range: (q_lo, q_hi) plotting grid extent — default (0.05, 0.95)
    """

    ndim = 2

    def __init__(
        self,
        copula,
        left_margin: tuple,
        right_margin: tuple,
        quantile_range: tuple[float, float] = (0.05, 0.95),
    ):
        _validate_margin(left_margin, "left_margin")
        _validate_margin(right_margin, "right_margin")

        self.copula = copula
        self.left_margin = _margin_to_dict(left_margin)
        self.right_margin = _margin_to_dict(right_margin)
        self._quantile_range = quantile_range
        self.ticks_nbr = 15

        self.new_seed()
        self._init_plot_grid()

    # ------------------------------------------------------------------
    # Plot grid
    # ------------------------------------------------------------------

    def _init_plot_grid(self, N: int = 200) -> None:
        q = self._quantile_range
        lm, rm = self.left_margin, self.right_margin
        x_range = lm["dist"].ppf(list(q), *lm["params"])
        y_range = rm["dist"].ppf(list(q), *rm["params"])
        self._x = np.linspace(x_range[0], x_range[1], N)
        self._y = np.linspace(y_range[0], y_range[1], N)
        self._X, self._Y = np.meshgrid(self._x, self._y)

    def _compute_pdf_grid(self) -> np.ndarray:
        return np.vectorize(lambda a, b: self.pdf([a, b]))(self._X, self._Y)

    # ------------------------------------------------------------------
    # RNG management
    # ------------------------------------------------------------------

    @property
    def seed(self) -> int:
        return self._seed

    def set_seed(self, key: int) -> None:
        self._seed = key
        self._rng = np.random.default_rng(self._seed)

    def new_seed(self) -> int:
        self.set_seed(secrets.randbits(128))
        return self._seed

    # ------------------------------------------------------------------
    # Joint distribution
    # ------------------------------------------------------------------

    def log_pdf(self, xy) -> float:
        """log f(x,y) = log f1(x) + log f2(y) + log c(F1(x), F2(y))."""
        lm, rm = self.left_margin, self.right_margin
        u1 = minmaxEPS(lm["dist"].cdf(xy[0], *lm["params"]))
        u2 = minmaxEPS(rm["dist"].cdf(xy[1], *rm["params"]))
        lf1 = float(lm["dist"].logpdf(xy[0], *lm["params"]))
        lf2 = float(rm["dist"].logpdf(xy[1], *rm["params"]))
        lc = np.log(max(float(self.copula.pdf(np.array([u1, u2]))), EPS))
        return float(lf1 + lf2 + lc)

    def pdf(self, xy) -> float:
        """Joint PDF at (x, y) in the original scale."""
        return float(np.exp(self.log_pdf(xy)))

    def cdf(self, xy) -> float:
        """Joint CDF: C(F1(x), F2(y))."""
        lm, rm = self.left_margin, self.right_margin
        u1 = minmaxEPS(lm["dist"].cdf(xy[0], *lm["params"]))
        u2 = minmaxEPS(rm["dist"].cdf(xy[1], *rm["params"]))
        return float(self.copula.cdf(np.array([u1, u2])))

    # ------------------------------------------------------------------
    # Conditional distribution  p(Y_cond | Y_obs = y_obs)
    # ------------------------------------------------------------------

    def _copula_uvs(
        self, y_cond: float, y_obs: float, which: str
    ) -> tuple[float, float, float]:
        """Return (u_obs, u_cond, log_f_cond) for the given conditioning side.

        which='left'  → y_obs is left,  y_cond is right
        which='right' → y_obs is right, y_cond is left
        """
        if which == "left":
            u_obs = minmaxEPS(
                self.left_margin["dist"].cdf(y_obs, *self.left_margin["params"])
            )
            u_cond = minmaxEPS(
                self.right_margin["dist"].cdf(y_cond, *self.right_margin["params"])
            )
            lf = float(
                self.right_margin["dist"].logpdf(y_cond, *self.right_margin["params"])
            )
        elif which == "right":
            u_obs = minmaxEPS(
                self.right_margin["dist"].cdf(y_obs, *self.right_margin["params"])
            )
            u_cond = minmaxEPS(
                self.left_margin["dist"].cdf(y_cond, *self.left_margin["params"])
            )
            lf = float(
                self.left_margin["dist"].logpdf(y_cond, *self.left_margin["params"])
            )
        else:
            raise ValueError("which must be 'left' or 'right'.")
        return u_obs, u_cond, lf

    def conditional_log_pdf(
        self, y_cond: float, y_obs: float, which: str = "left"
    ) -> float:
        """log p(y_cond | y_obs)  =  log f_cond(y_cond) + log c(u_obs, u_cond).

        which='left'  → condition on left  margin value, evaluate for right
        which='right' → condition on right margin value, evaluate for left
        """
        u_obs, u_cond, lf = self._copula_uvs(y_cond, y_obs, which)
        lc = np.log(max(float(self.copula.pdf(np.array([u_obs, u_cond]))), EPS))
        return float(lf + lc)

    def conditional_pdf(
        self, y_cond: float, y_obs: float, which: str = "left"
    ) -> float:
        """p(y_cond | y_obs)  =  f_cond(y_cond) · c(u_obs, u_cond)."""
        return float(np.exp(self.conditional_log_pdf(y_cond, y_obs, which)))

    def conditional_cdf(
        self, y_cond: float, y_obs: float, which: str = "left"
    ) -> float:
        """P(Y_cond ≤ y_cond | Y_obs = y_obs)  =  h(u_cond | u_obs).

        Uses the h-function h(v|u) = ∂C(u,v)/∂u provided by the copula.
        Valid for all symmetric copulas implemented here.
        """
        u_obs, u_cond, _ = self._copula_uvs(y_cond, y_obs, which)
        return float(self.copula.conditional_cdf(u_cond, u_obs))

    def conditional_law(self, y_obs: float, which: str = "left") -> ConditionalLaw:
        """Return a frozen conditional distribution p(Y_cond | Y_obs = y_obs).

        which='left'  → condition on left  margin, sample/evaluate the right
        which='right' → condition on right margin, sample/evaluate the left
        """
        return ConditionalLaw(self, y_obs, which)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, n: int = 1) -> np.ndarray:
        """Draw n iid samples from the joint distribution. Returns shape (n, 2)."""
        out = np.empty((n, 2))
        for i in range(n):
            y_left = self._sample_left()
            out[i, 0] = y_left
            out[i, 1] = self._sample_right_given_left(y_left)
        return out

    def sample_conditional(
        self, y_obs: float, which: str = "left", n: int = 1
    ) -> np.ndarray:
        """Draw n iid samples from p(Y_cond | Y_obs = y_obs). Returns shape (n,)."""
        if which == "left":
            return np.array([self._sample_right_given_left(y_obs) for _ in range(n)])
        elif which == "right":
            return np.array([self._sample_left_given_right(y_obs) for _ in range(n)])
        else:
            raise ValueError("which must be 'left' or 'right'.")

    def _sample_left(self) -> float:
        return float(
            self.left_margin["dist"].rvs(
                *self.left_margin["params"], random_state=self._rng
            )
        )

    def _sample_right_given_left(self, y_left: float) -> float:
        """Accept-reject: sample Y_right | Y_left = y_left."""
        u_left = minmaxEPS(
            self.left_margin["dist"].cdf(y_left, *self.left_margin["params"])
        )
        bound = self.copula.majorant(u_left)
        uv = np.array([u_left, -1.0])
        cpt = 0
        threshold = 0.0

        while cpt < _NBITERMAX:
            cpt += 1
            y_right = float(
                self.right_margin["dist"].rvs(
                    *self.right_margin["params"], random_state=self._rng
                )
            )
            u_right = minmaxEPS(
                self.right_margin["dist"].cdf(y_right, *self.right_margin["params"])
            )
            uv[1] = u_right
            threshold = self.copula.pdf(uv) / bound
            if self._rng.uniform() <= threshold:
                return y_right

        raise SamplingConvergenceError(
            f"Acceptance-rejection did not converge after {_NBITERMAX} iterations. "
            f"y_left={y_left}, majorant={bound:.4e}, last threshold={threshold:.4e}. "
            f"Check the copula majorant or increase _NBITERMAX."
        )

    def _sample_left_given_right(self, y_right: float) -> float:
        """Accept-reject: sample Y_left | Y_right = y_right.

        Valid for symmetric copulas (max_u c(u,v) = max_v c(u,v) = majorant(v)).
        """
        u_right = minmaxEPS(
            self.right_margin["dist"].cdf(y_right, *self.right_margin["params"])
        )
        bound = self.copula.majorant(u_right)
        uv = np.array([-1.0, u_right])
        cpt = 0
        threshold = 0.0

        while cpt < _NBITERMAX:
            cpt += 1
            y_left = float(
                self.left_margin["dist"].rvs(
                    *self.left_margin["params"], random_state=self._rng
                )
            )
            u_left = minmaxEPS(
                self.left_margin["dist"].cdf(y_left, *self.left_margin["params"])
            )
            uv[0] = u_left
            threshold = self.copula.pdf(uv) / bound
            if self._rng.uniform() <= threshold:
                return y_left

        raise SamplingConvergenceError(
            f"Acceptance-rejection did not converge after {_NBITERMAX} iterations. "
            f"y_right={y_right}, majorant={bound:.4e}, last threshold={threshold:.4e}. "
            f"Check the copula majorant or increase _NBITERMAX."
        )

    # ------------------------------------------------------------------
    # Parameter estimation (IFM two-step)
    # ------------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        data: np.ndarray,
        copula_class,
        left_family,
        right_family,
        copula_method: str = "tau",
        quantile_range: tuple[float, float] = (0.05, 0.95),
    ) -> "BivariateFitResult":
        """Two-step IFM fit: marginal MLE, then copula fit on the data.

        Step 1: each marginal distribution is MLE-fitted to its column.
        Step 2: the copula is fitted on the data (rank-based pseudo-obs,
                see :meth:`CopulaVirt.fit`).

        Parameters
        ----------
        data           : array-like, shape (n, 2)
        copula_class   : a CopulaVirt subclass (e.g. CopulaGaussian)
        left_family    : a scipy continuous distribution (e.g. scipy.stats.norm)
        right_family   : idem for the right margin
        copula_method  : 'tau' or 'mle' — passed to copula_class.fit
        quantile_range : forwarded to BivariateLaw plotting grid

        Returns
        -------
        BivariateFitResult
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError(f"data must be shape (n, 2), got {data.shape}.")
        n = data.shape[0]
        if n < 4:
            raise ValueError(f"At least 4 observations required, got {n}.")

        for fam, label in (
            (left_family, "left_family"),
            (right_family, "right_family"),
        ):
            if not all(
                hasattr(fam, m) for m in ("pdf", "cdf", "logpdf", "rvs", "ppf", "fit")
            ):
                raise ValueError(f"{label} must be a scipy continuous distribution.")

        left_params = tuple(left_family.fit(data[:, 0]))
        right_params = tuple(right_family.fit(data[:, 1]))

        copula_fit = copula_class.fit(data, method=copula_method)

        bivariate = cls(
            copula_fit.copula,
            (left_family, *left_params),
            (right_family, *right_params),
            quantile_range=quantile_range,
        )

        ll = float(np.sum([bivariate.log_pdf(data[i]) for i in range(n)]))

        return BivariateFitResult(
            bivariate=bivariate,
            copula_fit=copula_fit,
            left_params=left_params,
            right_params=right_params,
            log_likelihood=ll,
            n_obs=n,
            data=data,
        )

    @staticmethod
    def fit_best(
        data: np.ndarray,
        left_family,
        right_family,
        copula_families: list | None = None,
        copula_method: str = "tau",
        quantile_range: tuple[float, float] = (0.05, 0.95),
    ) -> list["BivariateFitResult"]:
        """Fit each candidate copula family with given margins, sorted by AIC.

        Parameters
        ----------
        data            : array-like, shape (n, 2)
        left_family     : a scipy continuous distribution
        right_family    : idem for the right margin
        copula_families : list of CopulaVirt subclasses, or None for a default set
                          (all available 1-parameter copulas except Product)
        copula_method   : 'tau' or 'mle'

        Returns
        -------
        list[BivariateFitResult]  sorted by AIC ascending. Failures are logged.
        """
        if copula_families is None:
            import importlib as _il
            from prg.copulas._base import CopulaEnum

            copula_families = []
            for _entry in CopulaEnum:
                if not _entry.value.AVAILABLE or not _entry.MODULE:
                    continue
                if _entry.CLASS_NAME == "CopulaProduct":  # τ fixé à 0, pas de fit utile
                    continue
                _mod = _il.import_module(_entry.MODULE)
                copula_families.append(getattr(_mod, _entry.CLASS_NAME))

        results: list = []
        for cls in copula_families:
            try:
                results.append(
                    BivariateLaw.fit(
                        data,
                        cls,
                        left_family,
                        right_family,
                        copula_method=copula_method,
                        quantile_range=quantile_range,
                    )
                )
            except Exception as e:
                logger.warning("BivariateLaw.fit with %s failed: %s", cls.__name__, e)
        results.sort(key=lambda r: r.aic)
        return results

    # ------------------------------------------------------------------
    # Representations
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        lp = ", ".join(f"{v:.2f}" for v in self.left_margin["params"])
        rp = ", ".join(f"{v:.2f}" for v in self.right_margin["params"])
        return (
            f"[\n  COPULA:       {self.copula}"
            f"\n  LEFT MARGIN:  {self.left_margin['dist_name']} ({lp})"
            f"\n  RIGHT MARGIN: {self.right_margin['dist_name']} ({rp})"
            "\n]"
        )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_pdf(self, plot_dir: str, prefix: str = "") -> None:
        z = self._compute_pdf_grid()
        fig, ax = plt.subplots(figsize=(6, 6), facecolor=facecolor)
        min_ = np.nanpercentile(z, 1.0)
        max_ = np.nanpercentile(z, 99.0)
        vticks = np.linspace(min_, max_, num=self.ticks_nbr)
        cs = ax.contourf(
            self._X, self._Y, z, vticks, antialiased=True, vmin=min_, vmax=max_
        )
        plt.colorbar(cs, ticks=vticks)
        ax.set_aspect("equal")
        ax.set_xlabel("Left margin")
        ax.set_ylabel("Right margin")
        tau = self.copula.params["tau_k"]
        cname = self.copula.copula_enum.value.LONG_NAME
        plt.suptitle(f"Joint PDF — {cname} (τ={tau})", y=0.85, fontsize=BIGGER_SIZE)
        plt.savefig(
            os.path.join(plot_dir, f"{prefix}2D_PdfLaw.png"),
            bbox_inches="tight",
            dpi=dpi,
            facecolor=facecolor,
        )
        plt.close()

    def plot_pdf_with_margins(
        self, plot_dir: str, title: str = "", prefix: str = ""
    ) -> None:
        z = self._compute_pdf_grid()

        fig = plt.figure(figsize=(6, 6), facecolor=facecolor)
        gs = gridspec.GridSpec(
            2,
            2,
            width_ratios=(4, 1),
            height_ratios=(1, 4),
            left=0.1,
            right=0.9,
            bottom=0.1,
            top=0.9,
            wspace=0.05,
            hspace=0.05,
        )
        ax = fig.add_subplot(gs[1, 0])
        ax_margin_left = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_margin_right = fig.add_subplot(gs[1, 1], sharey=ax)

        min_ = np.nanpercentile(z, 1.0)
        max_ = np.nanpercentile(z, 99.0)
        vticks = np.linspace(min_, max_, num=self.ticks_nbr)
        ax.contourf(self._X, self._Y, z, vticks, antialiased=True, vmin=min_, vmax=max_)
        ax.set_xlabel("Left margin")
        ax.set_ylabel("Right margin")

        ax_margin_left.tick_params(axis="x", labelbottom=False)
        ax_margin_right.tick_params(axis="y", labelleft=False)

        lparams, rparams = self.left_margin["params"], self.right_margin["params"]
        ax_margin_left.plot(
            self._x,
            self.left_margin["dist"].pdf(self._x, *lparams),
            "b-",
            lw=2,
            alpha=0.6,
        )
        ax_margin_right.plot(
            self.right_margin["dist"].pdf(self._y, *rparams),
            self._y,
            "b-",
            lw=2,
            alpha=0.6,
        )

        lname = ", ".join(f"{v:.2f}" for v in lparams)
        rname = ", ".join(f"{v:.2f}" for v in rparams)
        ax_margin_left.text(
            0.5,
            0.9,
            f"{self.left_margin['dist_name']} ({lname})",
            transform=ax_margin_left.transAxes,
            va="center",
            ha="center",
        )
        ax_margin_right.text(
            0.9,
            0.5,
            f"{self.right_margin['dist_name']} ({rname})",
            transform=ax_margin_right.transAxes,
            va="center",
            ha="center",
            rotation=270,
        )

        if not title:
            tau = self.copula.params["tau_k"]
            cname = self.copula.copula_enum.value.LONG_NAME
            title = f"Copula: {cname} (τ={tau})"
        fig.suptitle(title, y=0.95, fontsize=BIGGER_SIZE)
        plt.savefig(
            os.path.join(plot_dir, f"{prefix}2D_PdfLawWithMarginPdfs.png"),
            bbox_inches="tight",
            dpi=dpi,
            facecolor=facecolor,
        )
        plt.close()


# ---------------------------------------------------------------------------
# IFM fit result + GoF + bootstrap helpers
# ---------------------------------------------------------------------------


def _empirical_joint_cdf(data: np.ndarray, points: np.ndarray) -> np.ndarray:
    """F_n(x,y) = (1/n) Σ 1{X_i ≤ x, Y_i ≤ y} evaluated at given points."""
    leq_x = data[:, 0][None, :] <= points[:, 0][:, None]
    leq_y = data[:, 1][None, :] <= points[:, 1][:, None]
    return (leq_x & leq_y).mean(axis=1)


def _joint_cvm_statistic(data: np.ndarray, bivariate: "BivariateLaw") -> float:
    """Cramér-von Mises statistic on the joint: Σ (F_n(x,y) − F_θ(x,y))²."""
    Fn = _empirical_joint_cdf(data, data)
    Ftheta = np.array([bivariate.cdf(data[i]) for i in range(data.shape[0])])
    return float(np.sum((Fn - Ftheta) ** 2))


@dataclass
class BivariateBootstrapCI:
    """Returned by :meth:`BivariateFitResult.bootstrap_ci`."""

    tau_k: tuple[float, float]
    left_params: list[tuple[float, float]]
    right_params: list[tuple[float, float]]
    alpha: float
    B: int
    n_valid: int

    def __repr__(self) -> str:
        lp = ", ".join(f"[{lo:.3f},{hi:.3f}]" for lo, hi in self.left_params)
        rp = ", ".join(f"[{lo:.3f},{hi:.3f}]" for lo, hi in self.right_params)
        return (
            f"BivariateBootstrapCI({100*(1-self.alpha):.0f}%, "
            f"B={self.n_valid}/{self.B}, "
            f"tau_k=[{self.tau_k[0]:.3f},{self.tau_k[1]:.3f}], "
            f"left=[{lp}], right=[{rp}])"
        )


@dataclass
class BivariateFitResult:
    """Returned by :meth:`BivariateLaw.fit`."""

    bivariate: BivariateLaw
    copula_fit: FitResult
    left_params: tuple
    right_params: tuple
    log_likelihood: float  # ∑ log f(x_i, y_i) on the original data
    n_obs: int
    data: np.ndarray  # (n, 2) original data, kept for GoF / bootstrap

    @property
    def n_params(self) -> int:
        return len(self.left_params) + len(self.right_params) + self.copula_fit.n_params

    @property
    def aic(self) -> float:
        return 2.0 * self.n_params - 2.0 * self.log_likelihood

    @property
    def bic(self) -> float:
        return self.n_params * np.log(self.n_obs) - 2.0 * self.log_likelihood

    @property
    def aicc(self) -> float:
        """Corrected AIC for small samples: AIC + 2k(k+1)/(n−k−1)."""
        k, n = self.n_params, self.n_obs
        if n - k - 1 <= 0:
            return float("nan")
        return self.aic + 2.0 * k * (k + 1) / (n - k - 1)

    @property
    def hqc(self) -> float:
        """Hannan-Quinn: 2k·log(log n) − 2·loglik."""
        return (
            2.0 * self.n_params * np.log(np.log(self.n_obs)) - 2.0 * self.log_likelihood
        )

    def __repr__(self) -> str:
        cop = self.bivariate.copula.copula_enum.value.LONG_NAME
        lm = self.bivariate.left_margin["dist_name"]
        rm = self.bivariate.right_margin["dist_name"]
        return (
            f"BivariateFitResult(copula={cop!r}, "
            f"left={lm!r}{self.left_params}, "
            f"right={rm!r}{self.right_params}, "
            f"loglik={self.log_likelihood:.2f}, "
            f"AIC={self.aic:.2f}, BIC={self.bic:.2f}, n={self.n_obs})"
        )

    # ------------------------------------------------------------------
    # Goodness-of-fit on the joint (margins + copula)
    # ------------------------------------------------------------------
    def gof_test(self, B: int = 100, seed: int | None = None) -> GoFResult:
        """Cramér-von Mises GoF test on the joint distribution.

        Tests H₀: data was generated by the fitted bivariate (margins + copula).
        Bootstrap resamples from the fitted BivariateLaw and refits at each
        iteration — heavier than the copula-only test in :class:`FitResult`,
        but penalises misspecified margins as well.
        """
        try:
            self.bivariate.copula.cdf([0.5, 0.5])
        except NotImplementedError:
            raise NotImplementedError(
                "GoF test requires the copula CDF; not available for "
                f"{self.bivariate.copula.__class__.__name__}."
            )
        rng = np.random.default_rng(seed)
        S_n = _joint_cvm_statistic(self.data, self.bivariate)

        cop_class = self.copula_fit.copula.__class__
        left_family = self.bivariate.left_margin["dist"]
        right_family = self.bivariate.right_margin["dist"]
        method = self.copula_fit.method

        stats = np.full(B, np.nan)
        for b in range(B):
            self.bivariate.set_seed(int(rng.integers(0, 2**31 - 1)))
            sample = self.bivariate.sample(n=self.n_obs)
            try:
                r_b = BivariateLaw.fit(
                    sample, cop_class, left_family, right_family, copula_method=method
                )
                stats[b] = _joint_cvm_statistic(sample, r_b.bivariate)
            except Exception as e:
                logger.debug("Bootstrap iter %d failed: %s", b, e)

        valid = ~np.isnan(stats)
        n_valid = int(valid.sum())
        p_value = float(np.mean(stats[valid] >= S_n)) if n_valid > 0 else float("nan")
        return GoFResult(
            statistic=S_n,
            p_value=p_value,
            B=B,
            n_valid_bootstrap=n_valid,
            bootstrap_stats=stats[valid],
        )

    # ------------------------------------------------------------------
    # Bootstrap CI on all parameters (margins + tau_k)
    # ------------------------------------------------------------------
    def bootstrap_ci(
        self, B: int = 500, alpha: float = 0.05, seed: int | None = None
    ) -> BivariateBootstrapCI:
        """Non-parametric bootstrap percentile CI on every fitted parameter.

        Resamples (X_i, Y_i) pairs with replacement, refits the entire
        BivariateLaw (margins + copula), and returns (alpha/2, 1-alpha/2)
        quantiles for τ_k and each marginal parameter.
        """
        rng = np.random.default_rng(seed)
        cop_class = self.copula_fit.copula.__class__
        left_family = self.bivariate.left_margin["dist"]
        right_family = self.bivariate.right_margin["dist"]
        method = self.copula_fit.method
        n_left = len(self.left_params)
        n_right = len(self.right_params)

        tau_boots = np.full(B, np.nan)
        left_boots = np.full((B, n_left), np.nan)
        right_boots = np.full((B, n_right), np.nan)

        for b in range(B):
            idx = rng.integers(0, self.n_obs, size=self.n_obs)
            sample = self.data[idx]
            try:
                r_b = BivariateLaw.fit(
                    sample, cop_class, left_family, right_family, copula_method=method
                )
                tau_boots[b] = r_b.copula_fit.tau_k
                left_boots[b] = r_b.left_params
                right_boots[b] = r_b.right_params
            except Exception as e:
                logger.debug("Bootstrap iter %d failed: %s", b, e)

        valid = ~np.isnan(tau_boots)
        if valid.sum() < 10:
            raise RuntimeError(
                f"Bootstrap failed: only {valid.sum()}/{B} valid replicates."
            )

        q_lo, q_hi = alpha / 2.0, 1.0 - alpha / 2.0
        tau_ci = (
            float(np.quantile(tau_boots[valid], q_lo)),
            float(np.quantile(tau_boots[valid], q_hi)),
        )
        left_ci = [
            (
                float(np.quantile(left_boots[valid, k], q_lo)),
                float(np.quantile(left_boots[valid, k], q_hi)),
            )
            for k in range(n_left)
        ]
        right_ci = [
            (
                float(np.quantile(right_boots[valid, k], q_lo)),
                float(np.quantile(right_boots[valid, k], q_hi)),
            )
            for k in range(n_right)
        ]
        return BivariateBootstrapCI(
            tau_k=tau_ci,
            left_params=left_ci,
            right_params=right_ci,
            alpha=alpha,
            B=B,
            n_valid=int(valid.sum()),
        )

    # ------------------------------------------------------------------
    # K-fold cross-validation log-likelihood
    # ------------------------------------------------------------------
    def cv_loglik(self, K: int = 5, seed: int | None = None) -> float:
        """K-fold CV log-likelihood for the full bivariate model.

        For each fold, both margins (MLE) and the copula are refitted on the
        training set, and ∑ log f(x,y) is accumulated on the held-out test set.
        Higher is better. Cleaner than AIC: penalises overfitting honestly.
        """
        if K < 2 or K > self.n_obs:
            raise ValueError(f"K must be in [2, n_obs], got {K} for n={self.n_obs}.")
        rng = np.random.default_rng(seed)
        idx = rng.permutation(self.n_obs)
        folds = np.array_split(idx, K)
        cop_class = self.copula_fit.copula.__class__
        left_family = self.bivariate.left_margin["dist"]
        right_family = self.bivariate.right_margin["dist"]
        method = self.copula_fit.method

        cv_ll = 0.0
        for k in range(K):
            test_idx = folds[k]
            train_idx = np.concatenate([folds[j] for j in range(K) if j != k])
            try:
                r_k = BivariateLaw.fit(
                    self.data[train_idx],
                    cop_class,
                    left_family,
                    right_family,
                    copula_method=method,
                )
                for i in test_idx:
                    # Clip −∞ (support violation) at a large negative penalty.
                    # Common when expon/triang's fitted location excludes a test point.
                    log_p = float(r_k.bivariate.log_pdf(self.data[i]))
                    cv_ll += log_p if np.isfinite(log_p) else -100.0
            except Exception as e:
                logger.warning("CV fold %d failed: %s", k, e)
                return float("nan")
        return float(cv_ll)

    # ------------------------------------------------------------------
    # Visual diagnostics
    # ------------------------------------------------------------------
    def plot_diagnostics(self, plot_dir: str, prefix: str = "") -> None:
        """6-panel diagnostic plot:
        (0,0) data scatter + fitted joint PDF contours
        (0,1) QQ plot — left margin
        (0,2) lower tail dependence  λ̂_L(u) vs fitted λ_L (on parametric pseudo-obs)
        (1,0) QQ plot — right margin
        (1,1) PP plot of the joint CDF  F_n vs F_θ
        (1,2) upper tail dependence  λ̂_U(u) vs fitted λ_U
        """
        cop_name = self.bivariate.copula.copula_enum.value.LONG_NAME
        n = self.n_obs
        fig, axes = plt.subplots(2, 3, figsize=(16, 10), facecolor=facecolor)
        fig.suptitle(
            f"Bivariate fit diagnostics  ({cop_name}, n={n})",
            fontsize=BIGGER_SIZE + 1,
            y=0.995,
        )

        # (0,0) — data over fitted joint PDF
        ax = axes[0, 0]
        Z = self.bivariate._compute_pdf_grid()
        vmax = float(np.nanpercentile(Z, 95))
        ax.contourf(
            self.bivariate._X,
            self.bivariate._Y,
            Z,
            levels=10,
            cmap="Blues",
            alpha=0.45,
            vmin=0,
            vmax=max(vmax, 1e-10),
        )
        ax.scatter(self.data[:, 0], self.data[:, 1], s=4, alpha=0.6, c="black")
        ax.set_title("Data + fitted joint PDF", fontsize=BIGGER_SIZE)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        pp = (np.arange(1, n + 1) - 0.5) / n

        # (0,1) — left QQ
        ax = axes[0, 1]
        lm = self.bivariate.left_margin
        sx = np.sort(self.data[:, 0])
        tx = lm["dist"].ppf(pp, *lm["params"])
        ax.scatter(tx, sx, s=6, alpha=0.6)
        lo, hi = float(min(tx.min(), sx.min())), float(max(tx.max(), sx.max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_title(f"Left margin QQ  ({lm['dist_name']})", fontsize=BIGGER_SIZE)
        ax.set_xlabel("Theoretical")
        ax.set_ylabel("Empirical")

        # (1,0) — right QQ
        ax = axes[1, 0]
        rm = self.bivariate.right_margin
        sy = np.sort(self.data[:, 1])
        ty = rm["dist"].ppf(pp, *rm["params"])
        ax.scatter(ty, sy, s=6, alpha=0.6)
        lo, hi = float(min(ty.min(), sy.min())), float(max(ty.max(), sy.max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_title(f"Right margin QQ  ({rm['dist_name']})", fontsize=BIGGER_SIZE)
        ax.set_xlabel("Theoretical")
        ax.set_ylabel("Empirical")

        # (1,1) — joint PP plot
        ax = axes[1, 1]
        try:
            self.bivariate.copula.cdf([0.5, 0.5])
            Fn = _empirical_joint_cdf(self.data, self.data)
            Ft = np.array([self.bivariate.cdf(self.data[i]) for i in range(n)])
            ax.scatter(Fn, Ft, s=6, alpha=0.6)
            ax.plot([0, 1], [0, 1], "r--", lw=1)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.set_xlabel("$F_n$ (empirical)")
            ax.set_ylabel(r"$F_\theta$ (fitted)")
            ax.set_title("Joint PP plot", fontsize=BIGGER_SIZE)
        except NotImplementedError:
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                "Joint CDF not available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        # Tail dependence — pseudo-obs via FITTED margin CDFs (parametric)
        u1 = lm["dist"].cdf(self.data[:, 0], *lm["params"])
        u2 = rm["dist"].cdf(self.data[:, 1], *rm["params"])
        uv = np.column_stack([u1, u2])

        u_low = np.linspace(0.02, 0.30, 25)
        u_high = np.linspace(0.70, 0.98, 25)
        lL_emp = _empirical_tail_dep(uv, u_low, "lower")
        lU_emp = _empirical_tail_dep(uv, u_high, "upper")
        lL_th, lU_th = self.bivariate.copula.tail_dependence()

        # (0,2) — lower tail
        ax = axes[0, 2]
        ax.plot(u_low, lL_emp, "o-", ms=4, label=r"Empirical $\hat\lambda_L(u)$")
        if not np.isnan(lL_th):
            ax.axhline(
                lL_th,
                color="red",
                ls="--",
                lw=1.2,
                label=rf"Fitted $\lambda_L = {lL_th:.3f}$",
            )
        ax.set_xlabel("u")
        ax.set_ylabel(r"$\lambda_L$")
        ax.set_title("Lower tail dependence", fontsize=BIGGER_SIZE)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9)

        # (1,2) — upper tail
        ax = axes[1, 2]
        ax.plot(u_high, lU_emp, "o-", ms=4, label=r"Empirical $\hat\lambda_U(u)$")
        if not np.isnan(lU_th):
            ax.axhline(
                lU_th,
                color="red",
                ls="--",
                lw=1.2,
                label=rf"Fitted $\lambda_U = {lU_th:.3f}$",
            )
        ax.set_xlabel("u")
        ax.set_ylabel(r"$\lambda_U$")
        ax.set_title("Upper tail dependence", fontsize=BIGGER_SIZE)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9)

        plt.tight_layout()
        plt.savefig(
            os.path.join(plot_dir, f"{prefix}BivariateDiagnostics.png"),
            bbox_inches="tight",
            dpi=dpi,
            facecolor=facecolor,
        )
        plt.close()


if __name__ == "__main__":
    import scipy as sc
    from pathlib import Path
    from prg.copulas import *  # noqa: F401, F403 — script de démo

    left_margin = (sc.stats.triang, 0.158, 0.0, 3.0)
    right_margin = (sc.stats.norm, 5.0, 0.5)

    plot_dir = Path("./data/Plots/2DLaws")
    plot_dir.mkdir(parents=True, exist_ok=True)

    copulas = [
        CopulaGaussian(tau_k=0.5),
        CopulaProduct(tau_k=0.0),
        CopulaFGM(tau_k=0.2),
        CopulaA12(tau_k=0.7),
        CopulaA14(tau_k=0.7),
        CopulaGH(tau_k=0.2),
        CopulaClayton(tau_k=0.2),
        CopulaCubSec(tau_k=0.1),
        CopulaStudent(tau_k=0.1),
    ]
    for cop in copulas:
        law = BivariateLaw(cop, left_margin, right_margin)
        law.set_seed(42)
        print(law)

        samples = law.sample(5)
        print(f"  joint sample (5x2):\n{samples}")

        cond_left = law.conditional_law(y_obs=1.5, which="left")
        cond_right = law.conditional_law(y_obs=5.0, which="right")
        print(f"  {cond_left}  → sample(3): {cond_left.sample(3)}")
        print(f"  {cond_right} → sample(3): {cond_right.sample(3)}")

        law.plot_pdf(plot_dir, f"{cop.copula_enum.SHORT_NAME}_")
        law.plot_pdf_with_margins(plot_dir, prefix=f"{cop.copula_enum.SHORT_NAME}_")
