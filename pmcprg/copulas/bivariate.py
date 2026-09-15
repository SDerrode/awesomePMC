if __name__ == "__main__":
    import sys
    import pathlib

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import logging
import os
import secrets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from pmcprg.numerics import minmaxEPS, EPS, ONE_MINUS_EPS
# Re-export for backward compatibility — was raised by the AR sampler that
# is now superseded by Rosenblatt inversion. Kept so existing user code
# that imports it from ``pmcprg.copulas.bivariate`` still works.
from pmcprg.exceptions import SamplingConvergenceError       # noqa: F401
from pmcprg.copulas._bivariate_fit import (   # noqa: F401  (re-export for public API)
    BivariateBootstrapCI,
    BivariateFitResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------
#: Maximum number of acceptance-rejection iterations when sampling a
#: conditional law via the majorant method. Reaching this cap raises
#: :class:`pmcprg.exceptions.SamplingConvergenceError`. The default of 80 000 is
#: comfortable for all 17 copulas in their valid τ range; tweak it (e.g.
#: ``pmcprg.copulas.bivariate.MAX_AR_ITER = 200_000``) if a custom copula has
#: a poor analytical majorant on some τ regime.
MAX_AR_ITER = 80_000


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
        """Draw n iid samples from the joint distribution. Returns shape (n, 2).

        Vectorised Rosenblatt:
            y_left  ~ F_left      (n at once via ``rvs(size=n)``)
            u_left  = F_left(y_left)
            w       ~ Uniform(0, 1)  (n at once)
            v       = inv_h_array(w, u_left)
            y_right = F_right⁻¹(v)
        """
        lm, rm = self.left_margin, self.right_margin
        y_left  = np.asarray(
            lm["dist"].rvs(*lm["params"], size=n, random_state=self._rng),
            dtype=float,
        )
        u_left  = np.clip(lm["dist"].cdf(y_left, *lm["params"]),
                          EPS, ONE_MINUS_EPS)
        ws      = self._rng.uniform(EPS, ONE_MINUS_EPS, n)
        v       = self.copula.inv_h_array(ws, u_left)
        y_right = np.asarray(rm["dist"].ppf(v, *rm["params"]), dtype=float)
        return np.column_stack((y_left, y_right))

    def sample_conditional(
        self, y_obs: float, which: str = "left", n: int = 1
    ) -> np.ndarray:
        """Draw n iid samples from p(Y_cond | Y_obs = y_obs). Returns shape (n,).

        Vectorised Rosenblatt — scales as O(n) numpy ops, not O(n) Python
        loop iterations.
        """
        if which == "left":
            obs_margin, cond_margin = self.left_margin, self.right_margin
        elif which == "right":
            obs_margin, cond_margin = self.right_margin, self.left_margin
        else:
            raise ValueError("which must be 'left' or 'right'.")

        u_obs = float(np.clip(
            obs_margin["dist"].cdf(y_obs, *obs_margin["params"]),
            EPS, ONE_MINUS_EPS,
        ))
        ws = self._rng.uniform(EPS, ONE_MINUS_EPS, n)
        v  = self.copula.inv_h_array(ws, np.full(n, u_obs))
        return np.asarray(cond_margin["dist"].ppf(v, *cond_margin["params"]),
                          dtype=float)

    def _sample_left(self) -> float:
        return float(
            self.left_margin["dist"].rvs(
                *self.left_margin["params"], random_state=self._rng
            )
        )

    def _sample_right_given_left(self, y_left: float) -> float:
        """Sample Y_right | Y_left = y_left via Rosenblatt inversion.

        Steps:
            u_left = F_left(y_left)
            w     ~ Uniform(EPS, 1-EPS)
            v     = h⁻¹_C(w | u_left)            (closed form when available)
            y_right = F_right⁻¹(v)

        Closed-form ``inv_h`` (Gaussian, Clayton, Frank) → O(1) per sample.
        Other families fall back to a Brent inversion of ``conditional_cdf``
        inside ``CopulaVirt.inv_h``.
        """
        u_left = minmaxEPS(
            self.left_margin["dist"].cdf(y_left, *self.left_margin["params"])
        )
        w = float(self._rng.uniform(EPS, ONE_MINUS_EPS))
        v = float(self.copula.inv_h(w, u_left))
        return float(
            self.right_margin["dist"].ppf(v, *self.right_margin["params"])
        )

    def _sample_left_given_right(self, y_right: float) -> float:
        """Sample Y_left | Y_right = y_right via Rosenblatt inversion.

        Valid for symmetric copulas, where ``c(u, v) = c(v, u)`` and the
        conditional CDF satisfies ``h(u | v) = h(v | u)`` after the
        argument swap. All copulas in this package are symmetric.
        """
        u_right = minmaxEPS(
            self.right_margin["dist"].cdf(y_right, *self.right_margin["params"])
        )
        w = float(self._rng.uniform(EPS, ONE_MINUS_EPS))
        v = float(self.copula.inv_h(w, u_right))
        return float(
            self.left_margin["dist"].ppf(v, *self.left_margin["params"])
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
            from pmcprg.copulas._base import CopulaEnum

            copula_families = []
            for _entry in CopulaEnum:
                if not _entry.value.AVAILABLE or not _entry.value.MODULE:
                    continue
                if _entry.value.CLASS_NAME == "CopulaProduct":  # τ fixé à 0, pas de fit utile
                    continue
                _mod = _il.import_module(_entry.value.MODULE)
                copula_families.append(getattr(_mod, _entry.value.CLASS_NAME))

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
        fig, ax = plt.subplots(figsize=(6, 6))
        min_ = np.nanpercentile(z, 1.0)
        max_ = np.nanpercentile(z, 99.0)
        vticks = np.linspace(min_, max_, num=self.ticks_nbr)
        # extend="both": the levels are clipped to the 1st/99th percentiles,
        # so without it the joint density's peak — and its floor — are drawn
        # as unfilled holes rather than as the extremes they are.
        cs = ax.contourf(
            self._X, self._Y, z, vticks, antialiased=True,
            vmin=min_, vmax=max_, extend="both"
        )
        # Every third level: labelling all 15 crowded the bar with 4-decimal
        # numbers. Matches the copula plots' tick density.
        cbar = plt.colorbar(cs, ticks=vticks[::3])
        cbar.set_label("joint density f(x, y)")
        ax.set_aspect("equal")
        ax.set_xlabel("Left margin")
        ax.set_ylabel("Right margin")
        tau = self.copula.params["tau_k"]
        cname = self.copula.copula_enum.value.LONG_NAME
        plt.suptitle(f"Joint PDF — {cname} (τ={tau})", y=0.85)
        plt.savefig(
            os.path.join(plot_dir, f"{prefix}2D_PdfLaw.png"),
            bbox_inches="tight",
        )
        plt.close()

    def plot_pdf_with_margins(
        self, plot_dir: str, title: str = "", prefix: str = ""
    ) -> None:
        z = self._compute_pdf_grid()

        fig = plt.figure(figsize=(6, 6))
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
        ax.contourf(self._X, self._Y, z, vticks, antialiased=True,
                    vmin=min_, vmax=max_, extend="both")
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
        fig.suptitle(title, y=0.95)
        plt.savefig(
            os.path.join(plot_dir, f"{prefix}2D_PdfLawWithMarginPdfs.png"),
            bbox_inches="tight",
        )
        plt.close()



if __name__ == "__main__":
    import scipy as sc
    from pathlib import Path
    from pmcprg.copulas import (
        CopulaA12, CopulaA14, CopulaClayton, CopulaCubSec, CopulaFGM,
        CopulaGaussian, CopulaGH, CopulaProduct, CopulaStudent,
    )

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
